from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

class CifarModel:
    def __init__(self):
        self.model = Sequential()
        
    def build_model(self):
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # Clasificador
        from tensorflow.keras.layers import Flatten, Dense
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))
        
    def summary(self):
        print("\n=== Arquitectura del Modelo ===")
        self.model.summary()
        
    def get_model(self):
        return self.model
        
    def compile(self, **kwargs):
        self.model.compile(**kwargs)
        
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)