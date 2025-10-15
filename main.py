import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
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

if __name__ == "__main__":
    main()
    import gradio as gr
    def run_training():
        import io
        import matplotlib.pyplot as plt
        from tensorflow.keras.utils import plot_model
        from PIL import Image
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
        # Ejemplos de entrenamiento
        plt.figure(figsize=(6,2))
        for i in range(3):
            idx = i
            plt.subplot(1, 3, i+1)
            plt.imshow(dataset.x_train[idx])
            plt.axis('off')
            plt.title(f"Clase {dataset.class_names[np.argmax(dataset.y_train[idx])]}")
        buf_imgs = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf_imgs, format='png')
        buf_imgs.seek(0)
        img_examples = Image.open(buf_imgs)

        # Diagrama del modelo
        buf_model = io.BytesIO()
        plot_model(model.model, to_file=buf_model, show_shapes=True, show_layer_names=True, dpi=80, expand_nested=False)
        buf_model.seek(0)
        try:
            img_model = Image.open(buf_model)
        except Exception:
            img_model = None

        # Entrenamiento
        history = model.fit(
            dataset.x_train,
            dataset.y_train,
            epochs=5,
            validation_split=0.1,
            batch_size=32,
            verbose=0
        )
        # Gráficas de accuracy y loss
        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        buf_graph = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf_graph, format='png')
        buf_graph.seek(0)
        img_graph = Image.open(buf_graph)

        # Precisión final
        final_acc = history.history['val_accuracy'][-1]
        return img_examples, img_model, img_graph, f"Precisión final estimada: {final_acc:.3f}"

    with gr.Blocks() as demo:
        gr.Markdown("# Entrenamiento CNN CIFAR-10 (5 epochs)")
        btn = gr.Button("Entrenar y mostrar resultados")
        with gr.Tabs():
            with gr.Tab("Fase de entrenamiento"):
                img1 = gr.Image(label="Imágenes de ejemplo", type="pil")
            with gr.Tab("Diagrama del modelo"):
                img2 = gr.Image(label="Diagrama del modelo", type="pil")
            with gr.Tab("Gráficas y precisión"):
                img3 = gr.Image(label="Accuracy y Loss", type="pil")
                acc_text = gr.Textbox(label="Precisión final estimada")
        btn.click(run_training, outputs=[img1, img2, img3, acc_text])
    demo.launch()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        gradio_interface()
    else:
        main()
