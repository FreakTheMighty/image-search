from __future__ import print_function

import tempfile
import sys
import shutil
import scipy
import os
import numpy as np
import caffe
from redis import Redis
from lshash import LSHash
from flask import Flask, request, jsonify, send_from_directory

from werkzeug import secure_filename

app = Flask(__name__, static_url_path='')

redis = Redis(host='redis', port=6379)
caffe.set_mode_cpu()

app.config['UPLOAD_FOLDER'] = '/code/images/'
FEATURE_LAYER = 'fc7'
MODEL_FILE = '/code/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '/code/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
MEAN = np.load('/code/bvlc_reference_caffenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

NET = caffe.Classifier(
       MODEL_FILE, PRETRAINED, 
       mean = MEAN,
       channel_swap = (2,1,0),
       raw_scale = 255,
       image_dims = (256, 256)
    )

DESC = {'prev': None}

HASH = LSHash(8, 4096, num_hashtables=12, storage_config={'redis': {'port': 6379, 'host': 'redis'}})

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello():
    redis.incr('hits')
    return 'Hello World!! I have been seen %s times.' % redis.get('hits')


@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory('images', path)

@app.route('/images', methods=['POST'])
def addAndQuery():
    attached_file = request.files['image']
    if attached_file and allowed_file(attached_file.filename):
        warning('filename', attached_file.filename)
        filename = secure_filename(attached_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        attached_file.save(file_path)
        attached_file.close()
        warning(file_path)

        input_image = caffe.io.load_image(file_path)
        prediction = NET.predict([input_image]).flatten()
        descriptor = NET.blobs[FEATURE_LAYER].data[0].flatten()
        warning('descriptor', descriptor)

        HASH.index(descriptor, extra_data={
            'image': file_path.replace(app.config['UPLOAD_FOLDER'],'')
        })

        nearest = HASH.query(descriptor, distance_func='true_euclidean',)
        return jsonify({'hashes': nearest})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
