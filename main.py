import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify, send_file
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from src.cifar_dataset import CifarDataset
from src.cifar_model import CifarModel
import threading


app = Flask(__name__)

# Global dataset and model (loaded on startup)
DATASET = None
MODEL = None
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Training / plots state
PLOTS = {'accuracy': None, 'loss': None}
FINAL_ACC = None
TRAIN_LOCK = threading.Lock()
TRAINING_IN_PROGRESS = False


def load_resources():
    global DATASET, MODEL
    DATASET = CifarDataset()
    DATASET.load_data()
    DATASET.normalize_pixels()
    DATASET.convert_to_onehot()

    MODEL = CifarModel()
    MODEL.build_model()
    MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/images50')
def images50():
    # create a grid of 50 images from DATASET.x_test
    if DATASET is None:
        # return a tiny placeholder image if dataset not ready
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Dataset not loaded', ha='center', va='center')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    imgs = DATASET.x_test[:50]
    # if imgs are floats in [0,1], convert to uint8 0-255 for consistent display
    try:
        if imgs.dtype == np.float32 or imgs.max() <= 1.0:
            disp = (imgs * 255).astype(np.uint8)
        else:
            disp = imgs.astype(np.uint8)
    except Exception:
        disp = imgs

    fig, axes = plt.subplots(5, 10, figsize=(10, 5))
    idx = 0
    for r in range(5):
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(disp[idx])
            ax.axis('off')
            idx += 1

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    # prevent aggressive caching in browser
    resp = send_file(buf, mimetype='image/png')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp


@app.route('/image/<int:idx>')
def single_image(idx):
    """Return a single test image by index as PNG."""
    if DATASET is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Dataset not loaded', ha='center', va='center')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    if idx < 0 or idx >= len(DATASET.x_test):
        return jsonify({'error': 'index out of range'}), 404

    img = DATASET.x_test[idx]
    try:
        if img.dtype == np.float32 or img.max() <= 1.0:
            disp = (img * 255).astype(np.uint8)
        else:
            disp = img.astype(np.uint8)
    except Exception:
        disp = img

    pil_img = Image.fromarray(disp)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    resp = send_file(buf, mimetype='image/png')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp


@app.route('/predict', methods=['POST'])
def predict():
    # accept an uploaded image file
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((32, 32))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = MODEL.model.predict(arr)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return jsonify({'class': CLASS_NAMES[pred_idx], 'confidence': confidence})


def _train_and_record(epochs=10):
    """Run training and store accuracy/loss plots in-memory."""
    global PLOTS, FINAL_ACC, TRAINING_IN_PROGRESS
    with TRAIN_LOCK:
        TRAINING_IN_PROGRESS = True
    try:
        history = MODEL.model.fit(
            DATASET.x_train, DATASET.y_train,
            epochs=epochs,
            batch_size=64,
            validation_data=(DATASET.x_test, DATASET.y_test),
            verbose=1
        )

        # accuracy plot
        fig1, ax1 = plt.subplots()
        ax1.plot(history.history.get('accuracy', []), label='train_acc')
        ax1.plot(history.history.get('val_accuracy', []), label='val_acc')
        ax1.set_title('Accuracy')
        ax1.legend()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png')
        plt.close(fig1)
        buf1.seek(0)

        # loss plot
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history.get('loss', []), label='train_loss')
        ax2.plot(history.history.get('val_loss', []), label='val_loss')
        ax2.set_title('Loss')
        ax2.legend()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        plt.close(fig2)
        buf2.seek(0)

        PLOTS['accuracy'] = buf1.getvalue()
        PLOTS['loss'] = buf2.getvalue()
        FINAL_ACC = history.history.get('val_accuracy', [])[-1] if len(history.history.get('val_accuracy', [])) > 0 else None
    finally:
        with TRAIN_LOCK:
            TRAINING_IN_PROGRESS = False


@app.route('/train_and_plots', methods=['POST'])
def train_and_plots():
    """Start training in background and return status."""
    global TRAINING_IN_PROGRESS
    if DATASET is None or MODEL is None:
        return jsonify({'error': 'resources not ready'}), 500
    if TRAINING_IN_PROGRESS:
        return jsonify({'status': 'training_in_progress'}), 202
    t = threading.Thread(target=_train_and_record, kwargs={'epochs': 10}, daemon=True)
    t.start()
    return jsonify({'status': 'training_started'}), 202


@app.route('/plot/<name>')
def plot_image(name):
    key = None
    if name.lower().startswith('accuracy'):
        key = 'accuracy'
    elif name.lower().startswith('loss'):
        key = 'loss'
    else:
        return jsonify({'error': 'unknown plot'}), 404

    data = PLOTS.get(key)
    if data is None:
        # return a small placeholder image
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No plot yet', ha='center', va='center')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    return send_file(io.BytesIO(data), mimetype='image/png')


if __name__ == '__main__':
    load_resources()
    # Train synchronously before starting the server (10 epochs)
    print('Iniciando entrenamiento inicial (10 epochs) — la aplicación estará disponible después de completar el entrenamiento...')
    _train_and_record(epochs=10)
    if FINAL_ACC is not None:
        print(f'Entrenamiento inicial completado — accuracy final (val): {FINAL_ACC:.4f}')
    else:
        print('Entrenamiento inicial completado')
    app.run(debug=True)
