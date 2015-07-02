import requests
import json
import os

url = 'http://192.168.59.103:5000/images'
file_name = 'test-images/image2.jpg'
multiple_files = [('image', (os.path.basename(file_name), open(file_name, 'rb'), 'image/jpg'))]
print(requests.post(url, files=multiple_files).text)



