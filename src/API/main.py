import os

from manager import Manager

from fastapi.responses import HTMLResponse


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nevi.json"

app = Manager()
app.listen()


@app.get("/", response_class=HTMLResponse)
async def home():
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


@app.get("/img_in_db/", response_class=HTMLResponse)
async def show_files():
    files = app.ref.get()
    images = {}
    # print(files)
    for hash_key, case_list in files.items():
        for dictionary in case_list:
            # print(files[hash_key])
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

