import cv2
import tensorflow as tf
import numpy as np

class Classifier:

    def __init__(self,model_path):
        print(model_path)
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()
    def predict(self,img):
        img_1 = img/255
        prediction = self.model.predict(np.expand_dims(img_1,axis=0))
        certainty = np.max(prediction)
        category = np.argmax(prediction)
        return category, certainty
