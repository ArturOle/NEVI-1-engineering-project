from dataclasses import dataclass
import os
import unittest
from manager import Manager
from firebase_admin import credentials, initialize_app


class MockEvent:
    event_type: str = "patch"
    path: str = "/F5D1C6B3-2B0E-424D-88B4-DA238E9F08A0/-NDO2TfUs6z7wYouF1O8"
    data: dict = {
        'image_url': 'https://media.istockphoto.com/photos/mole-picture-id514399089?k=20&m=514399089&s=612x612&w=0&h=-qhWKCmJWJwTiGNlGKxuhsRps1c_l89K7Q3QxZo5n3g=',
        'diagnose': 'false',
    }


# def setUp(self) -> None:
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nevi.json"
#     cred = credentials.Certificate("nevi.json")
#     self.fire = initialize_app(cred)
#     self.event = MockEvent()

# def test_listner_on_patch(self):
#     app = Manager()
#     app.listner(self.event)


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nevi.json"
    event = MockEvent()
    app = Manager()
    app.listner(event)