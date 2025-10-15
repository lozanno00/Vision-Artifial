def analyze_performance(history, test_accuracy):
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]

    print("\n--- Análisis de rendimiento ---")
    print(f"Precisión final en entrenamiento: {train_acc:.2f}")
    print(f"Precisión final en validación: {val_acc:.2f}")
    print(f"Precisión final en test: {test_accuracy:.2f}")

    if train_acc > val_acc + 0.05:
        print("Posible sobreajuste detectado. Sugerencias:")
        print("- Añadir regularización (Dropout, L2)")
        print("- Data augmentation")
        print("- Reducir la complejidad del modelo")
    elif val_acc > train_acc:
        print("Posible subajuste detectado. Sugerencias:")
        print("- Añadir más capas o neuronas")
        print("- Entrenar más épocas")
        print("- Revisar la calidad de los datos")
    else:
        print("El modelo parece generalizar bien. Mejoras potenciales:")
        print("- Probar arquitecturas más profundas")
        print("- Ajustar hiperparámetros")
        print("- Más datos de entrenamiento")
