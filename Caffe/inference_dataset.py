import numpy as np
import caffe

from tools import *

# Config
cwd = '/workspace/semantic_segmentation'
experiment = cwd + '/External_repo/fcn.berkeleyvision.org/voc-fcn8s'
model_path = experiment + '/deploy.prototxt'
weights_path = experiment + '/fcn8s-heavy-pascal.caffemodel'
test_image_path = cwd + '/test_data/gato2.jpg'
output_path = test_image_path + '.npy'
dataset_path = cwd + '/Dataset/Pascal_voc_2012/VOCdevkit (2)/VOC2012'
dataset_names_path = dataset_path + '/ImageSets/Segmentation/val.txt'
dataset_images_path = dataset_path + '/JPEGImages'
experiment_path = cwd + '/Caffe/Results'
extension = '.jpg'
extension_result = '.npy'

network_size = (500, 500)

# Prepare Caffe and load the network
caffe.set_mode_cpu()
net = caffe.Net(model_path, caffe.TEST, weights=weights_path)

# Load dataset paths and show some information
with open(dataset_names_path) as f:
	list_image_names = f.readlines()

list_image_names = [x.strip() for x in list_image_names] # Removing \n

total_images = len(list_image_names)

print("Dataset total images: %d. \nLimiting the analysis to the first 100." % (total_images))

# Limit the analysis to the first 100 images of the dataset
list_image_names = list_image_names[:100]
total_images = len(list_image_names)

raw_input("Press Enter to continue...")


# Start the computation

list_elapsed_time = []

for num, image_name in enumerate(list_image_names):

	image_path = dataset_images_path + '/' + image_name + extension
	input_tensor = load_image_and_preprocess(image_path, network_size)
	output_tensor, elapsed = inference(net, input_tensor)
	list_elapsed_time.append(elapsed)

	# Save the result of the inference
	output_path = experiment_path + '/' + image_name + extension_result
	np.save(output_path, output_tensor)

	if True == True:
	#if num % 10: # TODO
		# Show inference time
		print("Sample %d of %d: inference took %f seconds" % (num+1, total_images, elapsed))

print("Yatta!") # Hiro Nakamura