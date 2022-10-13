from skimage import transform
from PIL import Image
from keras.applications.efficientnet_v2 import preprocess_input
from keras.models import load_model
import numpy as np


class Predictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, image):
        np_image = np.array(image).astype(float)
        np_image = transform.resize(np_image, (256, 256, 3))
        np_image = np.expand_dims(np_image, axis=0)
        np_image = preprocess_input(np_image)
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
        return map_results[np.argmax(results)]


if __name__ == "__main__":
    # with open(r"D:\Projects\thesis\grouped_images_by_severity\benign\ISIC_0024306.jpg", "r") as f:
    img = Image.open(r"D:\Projects\thesis\grouped_images_by_type\nv\ISIC_0034098.jpg")
    pred = Predictor(r"D:\Projects\thesis\model\NEVI_0.2.0.h5")
    pred = pred.predict(img)
    print(pred)
