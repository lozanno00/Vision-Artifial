from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot.png')
def plot_png():
    # Ejemplo: datos reales de predicción y accuracy
    # Reemplaza estos datos con los resultados reales de tu modelo
    epochs = [1, 2, 3, 4, 5]
    accuracy = [0.65, 0.72, 0.80, 0.85, 0.90]
    loss = [0.9, 0.7, 0.5, 0.3, 0.2]

    fig, ax1 = plt.subplots(figsize=(4,3))
    ax1.plot(epochs, accuracy, 'g-o', label='Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_title("Accuracy y Loss por Epoch")

    ax2 = ax1.twinx()
    ax2.plot(epochs, loss, 'r-s', label='Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    output = io.BytesIO()
    fig.savefig(output, format='png')
    output.seek(0)
    return send_file(output, mimetype='image/png')

@app.route('/train', methods=['POST'])
def train():
    # Aquí iría la lógica de entrenamiento
    return render_template('index.html', train_status="Entrenando imágenes...")

@app.route('/diagram.png')
def diagram_png():
    fig, ax = plt.subplots(figsize=(3,2))
    ax.axis('off')
    ax.text(0.5, 0.7, "Predicción", fontsize=14, ha='center')
    ax.text(0.5, 0.4, "Acierto: 92%", fontsize=12, ha='center')
    output = io.BytesIO()
    fig.savefig(output, format='png', bbox_inches='tight')
    output.seek(0)
    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
