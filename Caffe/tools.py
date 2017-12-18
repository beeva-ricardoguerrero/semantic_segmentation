import scipy.misc
import numpy as np
import caffe
import time


def load_image_and_preprocess(image_path, network_size):

	# load image and prepare as a single input batch for Caffe
	im = scipy.misc.imread(image_path)

	# Adapt the size to the network
	im = scipy.misc.imresize(im, network_size)

	# Adapt to match the tensor expected shape
	tensor = im[np.newaxis, :, :, :]
	tensor = np.rollaxis(tensor, 3, 1)

	return tensor


def inference(net, tensor_input):

	net.blobs['data'].reshape(*tensor_input.shape)
	net.blobs['data'].data[...] = tensor_input

	t1 = time.time()
	res = net.forward()
	t2 = time.time()

	elapsed = t2-t1

	return np.squeeze(res['score']), elapsed # res['score'] contains the output tensor


def load_gt(gt_path, network_size):

	# load ground truth images with it's specific format
	# VERY IMPORTANT TO LOAD GT THIS WAY. OTHERWISE THE VALUES WON'T BE CORRECT

	im =scipy.misc.imread(image_path, mode='P') # Split indexes from palette
	
	return im

def adapt_prediction(predicted, gt_shape):

	if len(predicted.shape) == 3: # We need to collapse probabilities
		predicted = np.argmax(predicted, 2)

	if not np.all(predicted.shape == gt_shape):

		# Adapt the size to the ground truth
		predicted = scipy.misc.imresize(predicted, gt_shape, interp='nearest')


# The following code is based on
# https://github.com/hszhao/PSPNet/blob/master/evaluation/evaluationCode/pixelAccuracy.m

# TODO check in the case of value 255 in gt
def single_pixelAccuracy(predicted, ground_truth):

	# Just to make sure
	assert predicted.shape == ground_truth.shape

	pixel_labeled = np.sum(predicted>0)
	pixel_correct = np.sum( (predicted == ground_truth) * (predicted > 0) )
	pixel_accuracy = pixel_correct / float(pixel_labeled)


def dataset_pixelAccuracy(ls_predicted, ls_ground_truth):
	
	eps = np.finfo(float).eps

	pixel_accuracy = [None]*len(ls_predicted)
	pixel_correct = [None]*len(ls_predicted)
	pixel_labeled = [None]*len(ls_predicted)

	for i in range(len(predicted)):

		pred = np.load(predicted[i])
		gt = load_gt(ground_truth[i])
		pred = adapt_prediction(pred, gt.shape)

		pixel_accuracy[i], pixel_correct[i], pixel_labeled[i] = \
		single_pixelAccuracy(pred, ground_truth[i])

	mean_pixel_accuracy = sum(pixel_correct)/float(eps + sum(pixel_labeled));

	return mean_pixel_accuracy