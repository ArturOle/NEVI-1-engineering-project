import shutil
import requests

url = 'https://firebasestorage.googleapis.com:443/v0/b/nevi-59237.appspot.com/o/images%2Fposts%2FF5D1C6B3-2B0E-424D-88B4-DA238E9F08A0%2F2022-10-02T13:57:09Z.jpg?alt=media&token=2a024bc4-8ad8-4307-829c-64ba296a1ee0'
image = requests.get(url, stream=True)
print(f"Image::  {image.raw}")
with open("image.jpg", 'wb') as f:
    shutil.copyfileobj(image.raw, f)
