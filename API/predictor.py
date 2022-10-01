# from PIL import Image
import numpy as np
from skimage import transform

from keras.models import load_model


class Predictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, image):
        np_image = np.array(image).astype('float32')/255
        np_image = transform.resize(np_image, (256, 256, 3))
        np_image = np.expand_dims(np_image, axis=0)
        result = self.model.predict(np_image)
        return result
