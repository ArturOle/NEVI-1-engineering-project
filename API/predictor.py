from skimage import transform
from PIL import Image
from keras.applications.efficientnet_v2 import preprocess_input
from keras.models import load_model
import numpy as np
from julia import Main
from os import environ, path
from pathlib import Path, PurePath
import logging
import shutil

ROOT = Path(__file__).resolve().parents[1]


class Predictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.julia = Main
        self.julia.include(str(ROOT/"image_cropper"/"image_cropper.jl"))

    def predict(self, image: Image):
        image = self.process(image)
        np_image = self.prepare_for_prediction(image)
        result = self.model.predict(np_image)[0]
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
            image.save("API/staged.png")
            image = self.julia.process("staged.png", "API")
            image = Image.open("API/staged.png")
        except Exception as err:
            logging.error(err)
        finally:
            return image

    def map_results(self, results: list):
        map_results = {
            0: "Actinic keratoses and intraepithelial carcinoma / Bowen's diesease",
            1: "Basal cell carcinoma",
            2: "Benign keratosis-like lesions",
            3: "Dermatofibroma",
            4: "Melanoma",
            5: "Melanocytic nevi",
            6: "Vascular lesions"
        }
        print(map_results[np.argmax(results)])
        return map_results[np.argmax(results)]


if __name__ == "__main__":
    img = Image.open(r"D:\Projects\thesis\data\grouped_images_by_type\akiec\ISIC_0010512.jpg")
    pred = Predictor(r"D:\Projects\thesis\src\model\NEVI_0.2.0.h5")
    pred = pred.predict(img)
    # print(pred)
