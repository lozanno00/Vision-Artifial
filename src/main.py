from cifar_dataset import CifarDataset
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
    else:
        print("\nAdvertencia: El dataset necesita revisión")
    
    dataset.show_examples(num_examples=3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()