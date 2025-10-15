def evaluate_and_report(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    return loss, accuracy
