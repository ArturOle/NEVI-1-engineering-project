# Built-in
import logging
import requests

# FastAPI
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

# Firebase
from firebase_admin import (
    db,
    credentials,
    initialize_app
)

# Tensorflow
from predictor import Predictor


class Manager(FastAPI):
    _path = "/posts"
    _url = "https://nevi-59237-default-rtdb.europe-west1.firebasedatabase.app"
    _cert = "NEVI.json"
    _firebase_app = None

    _bucket = None
    _templates = None

    def __init__(self) -> None:
        super().__init__(debug=False)
        logging.basicConfig(
            format='%(levelname)s:%(message)s',
            level=logging.INFO
        )
        self._log = logging.getLogger(__name__)
        self.ref = None
        self.last_diagnosis = None

    @property
    def firebase_app(self):
        if not self._firebase_app:
            cred = credentials.Certificate(self._cert)
            self._firebase_app = initialize_app(cred)
        return self._firebase_app

    @property
    def templates(self):
        if not self._templates:
            self._templates = Jinja2Templates(directory="templates")
        return self._templates

    def listen(self):
        self.ref = db.reference(
            path=self._path,
            url=self._url,
            app=self.firebase_app
        )
        self.ref.listen(self.listner)

    def listner(self, event):
        if event.event_type == "patch":
            image = event.data.get("image_url", None)
            if image:
                image = requests.get(image, stream=True)
                # print(image)
                prediction = Predictor().predict(image)
                db.reference(
                    path=''.join(["/", self._path, event.path, "/diagnose"]),
                    url=self._url, app=self.firebase_app
                ).set(prediction)
                self.last_diagnosis = prediction

    def home(self):
        return self.templates.TemplateResponse("home.html")
