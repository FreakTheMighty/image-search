from __future__ import print_function

from flask import Flask, request, jsonify
from redis import Redis
import caffe
import numpy as np
import os
import tempfile
import sys
from lshash import LSHash
import shutil
import scipy

app = Flask(__name__)
redis = Redis(host='redis', port=6379)

caffe.set_mode_cpu()

FEATURE_LAYER = 'fc7'
MODEL_FILE = '/code/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '/code/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
MEAN = np.load('/code/bvlc_reference_caffenet/ilsvrc_2012_mean.npy').mean(1).mean(1)

NET = caffe.Classifier(
       MODEL_FILE, PRETRAINED, 
       mean = MEAN,
       channel_swap = (2,1,0),
       raw_scale = 255,
       image_dims = (256, 256)
   )

DESC = {'prev': None}

HASH = LSHash(8, 256, num_hashtables=12, storage_config={'redis': {'port': 6379, 'host': 'redis'}})

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

@app.route('/')
def hello():
    redis.incr('hits')
    return 'Hello World!! I have been seen %s times.' % redis.get('hits')

@app.route('/images', methods=['POST'])
def addAndQuery():
    data = request.get_json()
    temp_file = tempfile.NamedTemporaryFile('wb', suffix='.png', delete=False)
    temp_file.write(data['imageData'].decode('base64'))
    temp_file.close()
    warning(temp_file.name);
    input_image = caffe.io.load_image(temp_file.name)
    shutil.copy(temp_file.name, './images/out.png')
    prediction = NET.predict([input_image]).flatten()
    descriptor = NET.blobs[FEATURE_LAYER].data[0].flatten()[0::16]
    #descriptor = descriptor[0::4]
    warning('descriptor', descriptor)

    nearest = HASH.query(descriptor, distance_func='true_euclidean');
    HASH.index(descriptor, extra_data=data['words']);
    warning('nearest', nearest)
    if (DESC['prev'] is not None):
      dist = scipy.spatial.distance.euclidean(DESC['prev'],descriptor)
      warning('distance', dist);
    DESC['prev'] = descriptor

    return jsonify({'hashes': nearest})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
