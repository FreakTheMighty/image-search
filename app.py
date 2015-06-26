from flask import Flask, request
from redis import Redis
import caffe
import numpy as np
import os
import tempfile
from lshash import LSHash

app = Flask(__name__)
redis = Redis(host='redis', port=6379)

caffe.set_mode_cpu()

FEATURE_LAYER = 'fc7'
MODEL_FILE = '/code/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '/code/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
MEAN = np.load('/code/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)

NET = caffe.Classifier(
       MODEL_FILE, PRETRAINED, 
       mean = MEAN,
       channel_swap = (2,1,0),
       raw_scale = 255,
       image_dims = (256, 256)
   )

HASH = LSHash(8, 4096, storage_config={'redis': {port: 6396}})

@app.route('/')
def hello():
    redis.incr('hits')
    return 'Hello World!! I have been seen %s times.' % redis.get('hits')

@app.route('/images', methods=['POST'])
def addAndQuery():
    data = request.get_json()
    temp_file = tempfile.NamedTemporaryFile('wb', suffix='png')
    temp_file.write(imgData.decode('base64'))
    temp_file.close()
    input_image = caffe.io.load_image(temp_file.name)
    prediction = NET.predict([input_image])  
    descriptor = np.linalg.norm(net.blobs[layer].data.flatten())
    HASH.index(descriptor);

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
