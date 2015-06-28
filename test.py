import requests
import json



url = 'http://192.168.59.105:5000/images'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
data = {
    'imageData': open('./image2.txt').read(),
    'words': ['hello', 'world']
}


print requests.post(url, data=json.dumps(data), headers=headers).text


