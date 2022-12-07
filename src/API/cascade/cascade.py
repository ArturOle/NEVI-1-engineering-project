from keras.models import load_model
from dataclasses import dataclass
from numpy import argmax


ROOT_MODEL = r"D:\Projects\thesis\src\model\cascade\layer1\experimental_model_layer1.h5"
FIRST_LAYER_BCC_DF = r"D:\Projects\thesis\src\model\cascade\layer2\experimental_model_layer22.h5"
FIRST_LAYER_MEL_NV = r"D:\Projects\thesis\src\model\cascade\layer2\experimental_model_layer2_mel_nv.h5"
FIRST_LAYER_BKL_AKIEC = r"D:\Projects\thesis\src\model\cascade\layer2\experimental_model_layer2_bkl_akiec.h5"


@dataclass
class Models:
    root_model = load_model(ROOT_MODEL)
    first_layer_bcc_df_model = load_model(FIRST_LAYER_BCC_DF)
    first_layer_mel_nv_model = load_model(FIRST_LAYER_MEL_NV)
    first_layer_bkl_akiec_model = load_model(FIRST_LAYER_BKL_AKIEC)



class Cascade:
    def __init__(self):
        self.models = Models()

    def predict(self, prediction_data):
        prediction = self.map_results_root(
            self.models.root_model.predict(prediction_data)
        )
        match prediction:
            case 'BCC_DF':
                return self.map_results_bcc_df(
                    self.models.first_layer_bcc_df_model.predict(prediction_data)
                )
            case 'MEL_NV':
                return self.map_results_mel_nv(
                    self.models.first_layer_mel_nv_model.predict(prediction_data)
                )
            case 'BKL_AKIEC':
                return self.map_results_bkl_akiec(
                    self.models.first_layer_bkl_akiec_model.predict(prediction_data)
                )
            case 'VASC':
                return 'VASC'

    def map_results_root(self, results: list):
        map_results = {
            0: 'BCC_DF',
            1: 'MEL_NV',
            2: 'BKL_AKIEC',
            3: 'VASC'
        }
        print(map_results[argmax(results)])
        return map_results[argmax(results)]

    def map_results_bcc_df(self, results: list):
        map_results = {
            0: 'BCC',
            1: 'DF'
        }
        print(map_results[argmax(results)])
        return map_results[argmax(results)]

    def map_results_mel_nv(self, results: list):
        map_results = {
            0: 'MEL',
            1: 'NV'
        }
        print(map_results[argmax(results)])
        return map_results[argmax(results)]

    def map_results_bkl_akiec(self, results: list):
        map_results = {
            0: 'BKL',
            1: 'AKIEC'
        }
        print(map_results[argmax(results)])
        return map_results[argmax(results)]
