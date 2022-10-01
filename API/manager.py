# Built-in
import logging

# FastAPI
from fastapi import FastAPI

# Firebase
from firebase_admin import db


class Manager(FastAPI):
    _path = "/posts"
    _url = "https://nevi-59237-default-rtdb.europe-west1.firebasedatabase.app"

    def __init__(self) -> None:
        super().__init__(debug=False)
        logging.basicConfig(
            format='%(levelname)s:%(message)s',
            level=logging.INFO
        )
        self._log = logging.getLogger(__name__)
        self.ref = None

    def listen(self):
        self._ref = db.reference(
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
            print(event.event_data)
