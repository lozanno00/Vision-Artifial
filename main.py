import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.cifar_dataset import CifarDataset
from src.cifar_model import CifarModel
import matplotlib.pyplot as plt

def main():
    dataset = CifarDataset()
    dataset.load_data()
    dataset.print_dimensions()
    dataset.normalize_pixels()
    dataset.convert_to_onehot()
    dataset.validate_dataset()

    model = CifarModel()
    model.build_model()
    model.summary()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        epochs=5,
        validation_split=0.1,
        batch_size=32,
        verbose=1
    )

    dataset.show_examples(num_examples=3)
    plt.tight_layout()
    plt.show()

def gradio_interface():
    import gradio as gr
    def run_training():
        import io
        import matplotlib.pyplot as plt
        dataset = CifarDataset()
        dataset.load_data()
        dataset.normalize_pixels()
        dataset.convert_to_onehot()
        model = CifarModel()
        model.build_model()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        history = model.fit(
            dataset.x_train,
            dataset.y_train,
            epochs=5,
            validation_split=0.1,
            batch_size=32,
            verbose=0
        )
        # Plot training accuracy
        plt.figure()
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return gr.Image.update(value=buf)

    with gr.Blocks() as demo:
        gr.Markdown("# Entrenamiento CNN CIFAR-10 (5 epochs)")
        btn = gr.Button("Entrenar y mostrar accuracy")
        img = gr.Image(label="Curva de accuracy", type="pil")
        btn.click(run_training, outputs=img)
    demo.launch()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        gradio_interface()
    else:
        main()
