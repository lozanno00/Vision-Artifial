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
    
    dataset.show_examples(num_examples=3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()