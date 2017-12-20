import numpy as np

from tools import *


cwd = '/workspace/semantic_segmentation'
dataset_path = cwd + '/Dataset/Pascal_voc_2012/VOCdevkit (2)/VOC2012'
dataset_names_path = dataset_path + '/ImageSets/Segmentation/val.txt'
dataset_ground_truth_path = dataset_path + '/SegmentationClass'
predicted_path = cwd + "/Caffe/Results"
ext_images = '.jpg'
ext_gt = '.png'
ext_result = '.npy'

network_size = (500, 500)


# Load dataset paths and show some information

with open(dataset_names_path) as f:
	list_image_names = f.readlines()

list_image_names = [x.strip() for x in list_image_names] # Removing \n

total_images = len(list_image_names)

print("Dataset total images: %d. \nLimiting the analysis to the first 100." % (total_images))

# Limit the analysis to the first 100 images of the dataset
list_image_names = list_image_names[:100]
total_images = len(list_image_names)



# Make predictions' paths

list_prediction_path = []

for num, image_name in enumerate(list_image_names):
	prediction_path = predicted_path + '/' + image_name + ext_result
	list_prediction_path.append(prediction_path)


# Make ground truth's path

list_ground_truth_path = []

for num, image_name in enumerate(list_image_names):
	gt_path = dataset_ground_truth_path + '/' + image_name + ext_gt
	list_ground_truth_path.append(gt_path)


print("Read %d predictions and %d ground truths" %(len(list_prediction_path), len(list_ground_truth_path)))
assert len(list_prediction_path) == len(list_ground_truth_path), \
"Error. The number of predictions doesn't match the number of ground truths."



# Start the computation

mean_pixel_accuracy = dataset_pixelAccuracy(list_prediction_path, list_ground_truth_path)

print("Report: the mean pixel accuracy for the whole dataset (%d images) is %.3lf \%" % (total_images, mean_pixel_accuracy*100))