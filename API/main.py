import os

from manager import Manager

from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi import UploadFile

from firebase_admin import credentials, initialize_app
from google.cloud import storage


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nevi.json"
cred = credentials.Certificate("nevi.json")
fire = initialize_app(cred)
client = storage.Client()
bucket = client.get_bucket('nevi-59237.appspot.com')
client.list_buckets()

app = Manager(fire)
app.listen()


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


# @app.post("/firebase/post")
# async def firebase_post(file: Optional[UploadFile] = None):
#     image_blob = bucket.blob("")
#     file.filename = f"{uuid.uuid4()}.jpg"
#     image_blob = bucket.blob("images/"+file.filename)
#     contents = await file.read()
#     save_file(file.filename, contents)
#     image_blob.upload_from_filename(file.filename)  # Upload your image
#     os.remove(file.filename)
#     return {"filename": file.filename}


@app.get("/img_in_db/", response_class=HTMLResponse)
async def show_files():
    files = app.ref.get()
    images = {}
    print(files)
    for hash_key, case_list in files.items():
        for dictionary in case_list:
            print(files[hash_key])
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
