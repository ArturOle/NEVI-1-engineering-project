from dataclasses import dataclass
import os
import unittest
from manager import Manager
from firebase_admin import credentials, initialize_app


@dataclass
class MockEvent:
    event_type: str = "patch"
    path: str = "/F5D1C6B3-2B0E-424D-88B4-DA238E9F08A0/-NDO2TfUs6z7wYouF1O8"
    data: dict = {
        'image_url': 'https://firebasestorage.googleapis.com:443/v0/b/nevi-59237.appspot.com/o/images%2Fposts%2FF5D1C6B3-2B0E-424D-88B4-DA238E9F08A0%2F2022-10-02T14:16:08Z.jpg?alt=media&token=f654f4e8-af7b-46c5-93b6-5a3a98ee4660',
        'diagnose': 'false',
        'image_height': 414,
        'created_at': 1664720169.715472
    }


class TestManager(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nevi.json"
        cred = credentials.Certificate("nevi.json")
        self.fire = initialize_app(cred)
        self.event = MockEvent()

    def test_listner_on_patch(self):
        app = Manager(self.fire)
        app.listner(MockEvent)
