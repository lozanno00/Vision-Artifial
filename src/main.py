import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from cifar_dataset import CifarDataset
from cifar_model import CifarModel
import matplotlib.pyplot as plt

def main():
    dataset = CifarDataset()
    dataset.load_data()
    dataset.print_dimensions()
    
    print("\nAntes de la normalización:")
    dataset.verify_normalization()
    
    dataset.normalize_pixels()
    
    print("\nDespués de la normalización:")
    dataset.verify_normalization()
    
    print("\nAntes de la conversión one-hot:")
    print(f"Forma original de y_train: {dataset.y_train.shape}")
    print(f"Ejemplo de etiqueta original:", dataset.y_train[0])
    
    dataset.convert_to_onehot()
    dataset.verify_onehot()
    
    if dataset.validate_dataset():
        print("\nEl dataset está listo para ser usado en la CNN")
        
        print("\n=== Inicialización del Modelo ===")
        model = CifarModel()
        print("\nModelo vacío:")
        model.summary()
        
        print("\nConstruyendo el modelo CNN:")
        model.build_model()
        print("\nModelo con ambos bloques convolucionales:")
        model.summary()
        
        print("\nAnálisis de la reducción del tensor:")
        print("Entrada original: (32, 32, 3)")
        print("Después del primer bloque:")
        print("- Conv2D: (30, 30, 32)")
        print("- MaxPooling: (15, 15, 32)")
        print("Después del segundo bloque:")
        print("- Conv2D: (13, 13, 64)")
        print("- MaxPooling: (6, 6, 64)")
    else:
        print("\nAdvertencia: El dataset necesita revisión")
    
    dataset.show_examples(num_examples=3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()