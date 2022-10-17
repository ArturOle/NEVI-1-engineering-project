from skimage import transform
from PIL import Image
from keras.applications.efficientnet_v2 import preprocess_input
from keras.models import load_model
import numpy as np
import shutil


class Predictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, image):
        # with open("image1.jpg", 'wb') as f:
        #     shutil.copyfileobj(image, f)
        image = Image.open(image.raw)
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
            1: "Basal cell carcinoma",
            2: "Benign keratosis-like lesions",
            3: "Dermatofibroma",
            4: "Melanoma",
            5: "Melanocytic nevi",
            6: "Vascular lesions"
        }
        return map_results[np.argmax(results)]


if __name__ == "__main__":
    img = Image.open(r"D:\Projects\thesis\grouped_images_by_type\vasc\ISIC_0024904.jpg")
    pred = Predictor(r"D:\Projects\thesis\model\NEVI_0.2.0.h5")
    pred = pred.predict(img)
    print(pred)
