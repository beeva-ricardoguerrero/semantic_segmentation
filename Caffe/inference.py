import numpy as np
import scipy
import caffe
import time

# Vars
cwd = '/workspace/semantic_segmentation'
experiment = cwd + '/External_repo/fcn.berkeleyvision.org/voc-fcn8s'
model_path = experiment + '/deploy.prototxt'
weights_path = experiment + '/fcn8s-heavy-pascal.caffemodel'
test_image_path = cwd + '/test_data/gato2.jpg'
output_path = test_image_path + '.npy'

# Load the net, list its data and params, and filter an example image.
caffe.set_mode_cpu()
net = caffe.Net(model_path, caffe.TEST, weights=weights_path)

#print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

# load image and prepare as a single input batch for Caffe
im =scipy.misc.imread(test_image_path)

# Adapt the size to the network's
im = scipy.misc.imresize(im, (500, 500))

im_input = im[np.newaxis, :, :, :]
im_input = np.rollaxis(im_input, 3, 1)
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

t1 = time.time()
res = net.forward()
t2 = time.time()

res = res['score']

#squeezed = np.argmax(res['score'], 1).squeeze()

np.save(output_path, res)

# Show inference time
print("Inference took %f seconds" % (t2-t1))