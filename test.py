import requests
import json
import os

url = 'http://192.168.59.103:5000/images'
file_name = 'test-images/image6.jpg'
multiple_files = [('image', (os.path.basename(file_name), open(file_name, 'rb'), 'image/jpg'))]
print(json.loads(requests.post(url, files=multiple_files).text))



