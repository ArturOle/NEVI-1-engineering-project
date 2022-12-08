from skimage import transform
from PIL import Image
from keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
from julia import Main
from pathlib import Path
import logging

from cascade.cascade import Cascade


ROOT = Path(__file__).resolve().parents[2]


class Predictor:
    def __init__(self):
        self.model = Cascade()
        self.julia = Main
        self.julia.include(str(ROOT/"src"/"image_cropper"/"image_cropper.jl"))

    def predict(self, image: Image):
        image = self.process(image)
        np_image = self.prepare_for_prediction(image)
        result = self.model.predict(np_image)
        print(result)
        return self.map_results(result)

    def prepare_for_prediction(self, image: Image):
        np_image = np.array(image).astype(float)
        np_image = transform.resize(np_image, (256, 256, 3))
        np_image = np.expand_dims(np_image, axis=0)
        np_image = preprocess_input(np_image)
        return np_image

    def process(self, image: Image):
        try:
            image.save("src/API/staged.png")
            image = self.julia.process("staged.png", "src/API")
            image = Image.open("src/API/staged.png")
        except Exception as err:
            logging.error(err)
        finally:
            return image

    def map_results(self, results: list):
        map_results = {
            'AKIEC': "Actinic keratoses and intraepithelial carcinoma / Bowen's diesease",
            'BCC': "Basal cell carcinoma",
            'BKL': "Benign keratosis-like lesions",
            'DF': "Dermatofibroma",
            'MEL': "Melanoma",
            'NV': "Melanocytic nevi",
            'VASC': "Vascular lesions"
        }
        print(map_results[results])
        return map_results[results]


if __name__ == "__main__":
    img = Image.open(r"D:\Projects\thesis\data\secret_test_folder\mel\ISIC_0034172.jpg")
    pred = Predictor()
    pred = pred.predict(img)
