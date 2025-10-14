from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

class CifarDataset:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
    
    def normalize_pixels(self):
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
    def verify_normalization(self):
        print("Verificación de normalización:")
        print(f"Rango de valores x_train: [{self.x_train.min():.3f}, {self.x_train.max():.3f}]")
        print(f"Rango de valores x_test: [{self.x_test.min():.3f}, {self.x_test.max():.3f}]")
        print(f"Forma de x_train: {self.x_train.shape}")
        print(f"Forma de x_test: {self.x_test.shape}")
    
    def convert_to_onehot(self):
        self.y_train = to_categorical(self.y_train, num_classes=10)
        self.y_test = to_categorical(self.y_test, num_classes=10)
    
    def verify_onehot(self):
        print("\nVerificación de codificación one-hot:")
        print(f"Forma de y_train: {self.y_train.shape}")
        print(f"Forma de y_test: {self.y_test.shape}")
        print(f"Número de clases: {self.y_train.shape[1]}")
        print("Ejemplo de etiqueta one-hot:", self.y_train[0])
        
    def print_dimensions(self):
        print(f"Training data shape: {self.x_train.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        print(f"Test data shape: {self.x_test.shape}")
        print(f"Test labels shape: {self.y_test.shape}")
        
    def get_images_by_class(self, class_idx, num_examples=2):
        if len(self.y_train.shape) == 2:  # Si las etiquetas están en formato one-hot
            class_indices = np.where(self.y_train[:, class_idx] == 1)[0]
        else:  # Si las etiquetas están en formato numérico
            class_indices = np.where(self.y_train.reshape(-1) == class_idx)[0]
        selected_indices = np.random.choice(class_indices, size=num_examples, replace=False)
        return self.x_train[selected_indices]
    
    def show_examples(self, num_examples=2):
        plt.figure(figsize=(15, 10))
        for class_idx in range(10):
            images = self.get_images_by_class(class_idx, num_examples)
            for j, img in enumerate(images):
                plt.subplot(10, num_examples, class_idx * num_examples + j + 1)
                plt.imshow(img)
                plt.axis('off')
                if j == 0:
                    plt.ylabel(self.class_names[class_idx])