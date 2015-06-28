import caffe
import numpy as np

caffe.set_mode_cpu()

FEATURE_LAYER = 'fc7'
MODEL_FILE = '/code/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '/code/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
MEAN = np.load('/code/bvlc_reference_caffenet/ilsvrc_2012_mean.npy').mean(1).mean(1)

IMAGE_FILE_1 = '/code/images/image1.jpg'
IMAGE_FILE_2 = '/code/images/image3.jpg'

NET = caffe.Classifier(
       MODEL_FILE, PRETRAINED, 
       mean = MEAN,
       channel_swap = (2,1,0),
       raw_scale = 255,
       image_dims = (256, 256)
   )

scale = 16
input_image = caffe.io.load_image(IMAGE_FILE_1)
prediction = NET.predict([input_image]).flatten()
descriptor_1 = NET.blobs[FEATURE_LAYER].data[0].flatten()[0::scale]

input_image = caffe.io.load_image(IMAGE_FILE_2)
prediction = NET.predict([input_image]).flatten()
descriptor_2 = NET.blobs[FEATURE_LAYER].data[0].flatten()[0::scale]

dist = np.linalg.norm(descriptor_1-descriptor_2)
print(dist)
