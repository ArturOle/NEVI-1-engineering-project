# Built-in
import os

# FastAPI
from urllib.request import Request
from fastapi import FastAPI, UploadFile
from typing import Optional
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Firebase
# from firebase_admin import credentials, firestore, initialize_app, db, storage
from google.cloud import storage
from firebase import firebase
import os
# from google.cloud import storage
# from google.oauth2 import service_account
# from google.cloud import storage


class MetaData(BaseModel):
    data_1: str   = None
    size_x: float = None
    size_y: float = None


# You just get your CREDENTIALS on previous step
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="moledetector.json"  
db_url='https://moledetector.firebaseio.com'   # Your project url
firebase = firebase.FirebaseApplication(db_url,None)
client = storage.Client()
bucket = client.get_bucket('moledetector.appspot.com')
client.list_buckets()
# imageBlob = bucket.blob("/")
# imagePath = "path/to/dir/" + fileName  # Replace with your own path
# imageBlob = bucket.blob(fileName)
# imageBlob.upload_from_filename(imagePath)

# Firebase initialization
# cred = credentials.Certificate("moledetector.json")
# initialize_app(cred, {'storageBucket': 'moledetector.appspot.com'})
# firedb = storage.
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "moledetector.json"
# storage_client = storage.Client()

# # List all the buckets available (@ScottMcC code)
# for bucket in storage_client.list_buckets():
#     print(bucket)

# #or you can list it
# list(storage_client.list_buckets())

# FastAPI initialization
app = FastAPI()
templates = Jinja2Templates(directory="templates")

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
    for bucket in storage_client.list_buckets():
        buckets.append(bucket)
    return buckets

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
    """.format(files="\n".join(db))

@app.post("/images/")
async def create_upload_file(file: Optional[UploadFile] = None):
    # db.append(file.filename)
    return {"filename": file.filename}
