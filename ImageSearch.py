from __future__ import print_function

import tempfile
import sys
import shutil
import scipy
import os
import numpy as np
import json
import caffe
from redis import Redis
from lshash import LSHash
from rq import Queue
import time

redis = Redis(host='redis', port=6379)
queue = Queue(connection=redis)
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

HASH = LSHash(8, 4096, num_hashtables=2, storage_config={'redis': {'port': 6379, 'host': 'redis'}})

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def index_image(file_path):
    warning('Indexing image: ' + file_path)
    input_image = caffe.io.load_image(file_path)
    prediction = NET.predict([input_image])
    descriptor = NET.blobs[FEATURE_LAYER].data[0].flatten()
    warning('desctriptor', descriptor)
    HASH.index(descriptor, extra_data={
        'image': file_path
    })

def query_image(file_path):
    input_image = caffe.io.load_image(file_path)
    prediction = NET.predict([input_image])
    descriptor = NET.blobs[FEATURE_LAYER].data[0].flatten()
    return HASH.query(descriptor, distance_func='true_euclidean')
