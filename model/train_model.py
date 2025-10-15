def train_model(model, x_train, y_train, epochs=10, validation_split=0.1):
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_split=validation_split
    )
    return history
