# Built-in
import os
import logging

# FastAPI
from fastapi import FastAPI, UploadFile
from typing import Optional
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uuid

# Firebase
from firebase_admin import credentials, db, initialize_app
from google.cloud import storage

logging.basicConfig(
    format='%(levelname)s:%(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)


# FastAPI initialization
app = FastAPI(debug=False)
templates = Jinja2Templates(directory="templates")


# Firebase initialization
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nevi.json"
cred = credentials.Certificate("nevi.json")
fire = initialize_app(cred)
client = storage.Client()
bucket = client.get_bucket('nevi-59237.appspot.com')
client.list_buckets()
ref = db.reference(
    path="/posts",
    url="https://nevi-59237-default-rtdb.europe-west1.firebasedatabase.app"
)


class MetaData(BaseModel):
    data_1: str = None
    size_x: float = None
    size_y: float = None


def listner(event):
    log.info(
        'Data: event_type={} path={} other={}'.format(
            event.event_type,
            event.path,
            event.data
        )
    )
    if event.event_type == "patch":
        print(event.event_data)


def save_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)


@app.get("/", response_class=HTMLResponse)
async def welcome():
    return """
        <html>
        <head>
            <title>Item Details</title>
        </head>
        <body>
            <h1>Item ID: </h1>
        </body>
        </html>
    """


@app.get("/firebase/")
async def firebase_info():
    buckets = []
    for bucket in client.list_buckets():
        buckets.append(bucket)
    return str(buckets)


@app.post("/firebase/post")
async def firebase_post(file: Optional[UploadFile] = None):
    image_blob = bucket.blob("")
    file.filename = f"{uuid.uuid4()}.jpg"
    image_blob = bucket.blob("images/"+file.filename)
    contents = await file.read()
    save_file(file.filename, contents)
    image_blob.upload_from_filename(file.filename)  # Upload your image
    os.remove(file.filename)
    return {"filename": file.filename}


@app.get("/img_in_db/", response_class=HTMLResponse)
async def show_files():
    files = ref.get()
    images = {}

    for hash_key, case_list in files.items():
        for dictionary in case_list:
            images[dictionary] = files[hash_key][dictionary]["image_url"]

    data = [
        f"<h2>{hash_key}</h2>\n<img src={image}>"
        for hash_key, image in images.items()
    ]

    return """
        <html>
        <head>
            <title>Item Details</title>
        </head>
        <body>
            <h1>Files:</h1>
            {}
        </body>
        </html>
    """.format('\n'.join(data))


@app.post("/images/")
async def create_upload_file(file: Optional[UploadFile] = None):
    # db.append(file.filename)
    return {"filename": file.filename}


ref.listen(listner)
