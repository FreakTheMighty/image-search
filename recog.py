import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

IMAGE_FILE = 'images/image6.jpg'
IMAGE_FILE_1 = 'images/image9.jpg'


caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, 
        mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
        channel_swap=(2,1,0),
        raw_scale=255,
        image_dims=(256, 256))

input_image = caffe.io.load_image(IMAGE_FILE)
prediction = net.predict([input_image])  

layer = 'fc7'
print "shape", net.blobs[layer].data.shape
vec1 = np.linalg.norm(net.blobs[layer].data.flatten())

input_image_1 = caffe.io.load_image(IMAGE_FILE_1)
prediction_1 = net.predict([input_image_1])  
vec2 = np.linalg.norm(net.blobs[layer].data.flatten())

print prediction.shape
#vec1 = prediction.flatten()
#vec2 = prediction_1.flatten()

dist = np.linalg.norm(vec1-vec2)
print 'distance:', dist 


print 'predicted class1:', prediction[0].argmax()
print 'predicted class:', prediction_1[0].argmax()

