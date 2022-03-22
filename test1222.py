from __future__ import print_function
from imutils import paths
from scipy.io import loadmat
from skimage import io
from skimage import measure
import argparse
import dlib
import sys
import os
import cv2
import imutils
import numpy as np

array_of_img = []
def read_directory(directory_name):
	for filename in os.listdir(r"./"+directory_name):
		img = cv2.imread(directory_name + "/" + filename)
		array_of_img.append(img)

kernel = np.ones((5,5), np.uint8)
boxes = []
images = []

options = dlib.simple_object_detector_training_options()

read_directory("Image")
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to the output detector")
args = vars(ap.parse_args())
print("[INFO] gathering images and bounding boxes...")

for image in array_of_img:
	(B,G,R) = cv2.split(image)
	blurred = cv2.GaussianBlur(R, (7, 7), 0)

	(T, thresh1) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
	erosion = cv2.erode(thresh1, kernel, iterations = 10)
	thresh1 = cv2.dilate(erosion, kernel, iterations = 10)

	labels = measure.label(thresh1, neighbors=8, background=0)
	mask = np.zeros(thresh1.shape, dtype="uint8")

	for (i, label) in enumerate(np.unique(labels)):
		if label == 0:
			print()
			continue

		labelMask = np.zeros(thresh1.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)

		if numPixels > 120000:

			mask = cv2.add(mask, labelMask)
			cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)
			clone = image.copy()
			for c in cnts:
				(x, y, w, h) = cv2.boundingRect(c)
				cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
			mango = cv2.bitwise_and(image, image, mask=mask)
	if w*h < 400000:
			if x != 0:
				bb = [dlib.rectangle(left=int(x), top=int(y), right=int(x+w), bottom=int(y+h))]
				boxes.append(bb)
				images.append(image)

print(boxes)
print("[INFO] training detector...")
detector = dlib.train_simple_object_detector(images,boxes,options)
print("[INFO] dumping classifier to file...")
detector.save(args ["output"])
win = dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()