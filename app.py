from __future__ import print_function

import tempfile
import sys
import shutil
import scipy
import os
import numpy as np
import json
from redis import Redis
from lshash import LSHash
from flask import Flask, request, jsonify, send_from_directory
from rq import Queue

import ImageSearch

from werkzeug import secure_filename

app = Flask(__name__, static_url_path='')

redis = Redis(host='redis', port=6379)
queue = Queue(connection=redis)

app.config['UPLOAD_FOLDER'] = '/code/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

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
def send_image(path):
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
        nearest = ImageSearch.query_image(file_path)
        warning('sending to queue', file_path)
        queue.enqueue('ImageSearch.index_image', file_path)
        results = []
        for near in nearest:
            extra_data = json.loads(near[0])[1]
            distance = near[1]
            results.append((distance, extra_data))

        return jsonify({'nearest': results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
