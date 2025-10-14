from tensorflow.keras.models import Sequential

class CifarModel:
    def __init__(self):
        self.model = Sequential()
        
    def build_model(self):
        pass
        
    def summary(self):
        self.model.summary()
        
    def get_model(self):
        return self.model