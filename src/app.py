import gradio as gr
import numpy as np
from cifar_dataset import CifarDataset
from cifar_model import CifarModel
import matplotlib.pyplot as plt
import io
from PIL import Image

class CifarDemo:
    def __init__(self):
        self.dataset = CifarDataset()
        self.model = CifarModel()
        self.initialize()
    
    def initialize(self):
        self.dataset.load_data()
        self.dataset.normalize_pixels()
        self.dataset.convert_to_onehot()
        self.model.build_model()
    
    def get_dataset_info(self):
        info = []
        info.append("=== Información del Dataset ===")
        info.append(f"Imágenes de entrenamiento: {self.dataset.x_train.shape}")
        info.append(f"Etiquetas de entrenamiento: {self.dataset.y_train.shape}")
        info.append(f"Imágenes de prueba: {self.dataset.x_test.shape}")
        info.append(f"Etiquetas de prueba: {self.dataset.y_test.shape}")
        info.append("\nRango de valores:")
        info.append(f"Min: {self.dataset.x_train.min():.3f}, Max: {self.dataset.x_train.max():.3f}")
        return "\n".join(info)
    
    def get_model_summary(self):
        import io
        import sys
        
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        self.model.summary()
        
        sys.stdout = old_stdout
        return new_stdout.getvalue()
    
    def show_random_images(self, num_images=6):
        plt.figure(figsize=(12, 4))
        for i in range(num_images):
            idx = np.random.randint(0, len(self.dataset.x_train))
            plt.subplot(1, num_images, i + 1)
            img = self.dataset.x_train[idx]
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Clase {np.argmax(self.dataset.y_train[idx])}")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return Image.open(buf)

def create_interface():
    demo = CifarDemo()
    
    with gr.Blocks(title="CIFAR-10 CNN Demo") as interface:
        gr.Markdown("# CIFAR-10 CNN Demonstración")
        
        with gr.Tab("Dataset"):
            gr.Markdown("### Información del Dataset")
            dataset_info = gr.Textbox(value=demo.get_dataset_info(), label="Información")
            show_btn = gr.Button("Mostrar ejemplos aleatorios")
            image_output = gr.Image(type="pil")
            show_btn.click(fn=demo.show_random_images, outputs=image_output)
        
        with gr.Tab("Modelo"):
            gr.Markdown("### Arquitectura del Modelo")
            model_info = gr.Textbox(value=demo.get_model_summary(), label="Arquitectura")
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)