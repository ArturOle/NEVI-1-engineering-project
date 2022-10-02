# Built-in
import logging
import requests

# FastAPI
from fastapi import FastAPI

# Firebase
from firebase_admin import db

# Tensorflow
from predictor import Predictor


class Manager(FastAPI):
    _path = "/posts"
    _url = "https://nevi-59237-default-rtdb.europe-west1.firebasedatabase.app"
    _bucket = None

    def __init__(self, firebase) -> None:
        super().__init__(debug=False)
        logging.basicConfig(
            format='%(levelname)s:%(message)s',
            level=logging.INFO
        )
        self.firebase_app = firebase
        self._log = logging.getLogger(__name__)
        self.ref = None

    def listen(self):
        self.ref = db.reference(
            self._path,
            self._url
        )
        self.ref.listen(self.listner)

    def listner(self, event):
        self._log.info(
            'Data: event_type={} path={} other={}'.format(
                event.event_type,
                event.path,
                event.data
            )
        )
        if event.event_type == "patch":
            image = event.other.get("image_url", default=None)
            if image:
                image = requests.get(image)
                prediction = Predictor(r"D:\Projects\thesis\model\experimental_model_severity_full90prc_S5.h5").predict(image)
                db.reference(''.join([event.path, "is_diagnose_ready"])).set(prediction)
