import requests
import json



url = 'http://192.168.59.105:5000/images'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
data = {
    'imageData': open('./image7.txt').read(),
    'words': ['image 7']
}


print requests.post(url, data=json.dumps(data), headers=headers).text


