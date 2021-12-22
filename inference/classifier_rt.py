import cv2
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.preprocessing import image

class ClassifierRT:

    def __init__(self,model_path):
        self.saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']
    
    def predict(self,img):
        img_1 = img/255
        img_1 = np.expand_dims(img_1, axis=0)
        img_1 = img_1.astype('float32')
        img_1 = tf.constant(img_1)
        labeling = self.infer(img_1)
        #print(labeling)
        prediction = labeling['dense_28'].numpy()
        certainty = np.max(prediction)
        category = np.argmax(prediction)
        return category, certainty