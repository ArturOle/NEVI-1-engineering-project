from PIL import Image

import numpy as np
from skimage import transform

from keras.models import load_model


class Predictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, image):
        image = Image.open(image)
        np_image = np.array(image).astype('float32')/255
        np_image = transform.resize(np_image, (256, 256, 3))
        np_image = np.expand_dims(np_image, axis=0)
        result = self.model.predict(np_image)[0]
        print(result)
        return self.map_results(result)

    def map_results(self, results: list):
        map_results = {
            0: "Actinic keratoses and intraepithelial carcinoma / Bowen's diesease",
            1: "You die bro",
            2: "Benign keratosis-like lesions",
            3: "Dermatofibroma",
            4: "Melanoma",
            5: "Melanocytic nevi",
            6: "Vascular lesions"
        }
        return map_results[np.array([np.argmax(x) for x in results])]


if __name__ == "__main__":
    img = Image.open(r"D:\Projects\thesis\secret_test_folder\mel\ISIC_0034243.jpg")
    pred = Predictor(r"D:\Projects\thesis\model\NEVI_0.2.0.h5")
    pred = pred.predict(img)
    print(pred)
