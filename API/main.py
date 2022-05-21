# Built-in
import os

# FastAPI
from urllib.request import Request
from fastapi import FastAPI, UploadFile
from typing import Optional
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uuid

# Firebasews
from firebase_admin import credentials, initialize_app
from google.cloud import storage


class MetaData(BaseModel):
    data_1: str   = None
    size_x: float = None
    size_y: float = None


# Firebase initialization
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="moledetector.json"  
cred = credentials.Certificate("moledetector.json")
fire = initialize_app(cred)
client = storage.Client()
bucket = client.get_bucket('moledetector.appspot.com')
client.list_buckets()

# FastAPI initialization
app = FastAPI(debug=False)
templates = Jinja2Templates(directory="templates")

def save_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)

@app.get("/", response_class=HTMLResponse)
async def welcome():
    return """
        <html>
        <head>
            <title>Item Details</title>
            <link href="{{ url_for('static', path='/styles.css') }}" rel="stylesheet">
        </head>
        <body>
            <h1>Item ID: </h1>
        </body>
        </html>   
    """

@app.get("/firebase/")
async def firebase_info():
    buckets=[]
    for bucket in client.list_buckets():
        buckets.append(bucket)
    return str(buckets)

@app.post("/firebase/post")
async def firebase_post(file: Optional[UploadFile] = None):
    imageBlob = bucket.blob("")
    file.filename = f"{uuid.uuid4()}.jpg"
    imageBlob = bucket.blob("images/"+file.filename)
    contents = await file.read()
    save_file(file.filename, contents)
    imageBlob.upload_from_filename(file.filename) # Upload your image
    os.remove(file.filename)
    return {"filename": file.filename}

@app.get("/img_in_db/", response_class=HTMLResponse)
async def show_files():
    return """
        <html>
        <head>
            <title>Item Details</title>
        </head>
        <body>
            <h1>Files: {files}</h1>
        </body>
        </html>   
    """.format(files="\n".join(["a", "b"]))

@app.post("/images/")
async def create_upload_file(file: Optional[UploadFile] = None):
    # db.append(file.filename)
    return {"filename": file.filename}

