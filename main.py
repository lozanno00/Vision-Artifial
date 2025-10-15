from src.main import get_model_and_data
from model.compile_model import compile_and_summarize
from model.train_model import train_model
from model.plot_history import plot_history
from model.evaluate_model import evaluate_and_report
from model.performance_analysis import analyze_performance

def main():
    # Obtener modelo y datos
    model, x_train, y_train, x_test, y_test = get_model_and_data()

    # Compilar y mostrar resumen
    compile_and_summarize(model)

    # Entrenar el modelo
    history = train_model(model, x_train, y_train, epochs=10, validation_split=0.1)

    # Visualizar el aprendizaje
    plot_history(history)

    # Evaluar el rendimiento final
    loss, test_accuracy = evaluate_and_report(model, x_test, y_test)

    # Análisis de rendimiento
    analyze_performance(history, test_accuracy)

if __name__ == "__main__":
    main()
