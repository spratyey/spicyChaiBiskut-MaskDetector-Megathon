# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

import mtcnn

import mtcnn
from mtcnn.mtcnn import MTCNN
import cv2
import csv

# model = MTCNN()

import numpy as np

# face detection with mtcnn on a photograph
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

# use MTCNN to detct faces
detector = MTCNN()
def detect_and_predict_mask(frame, maskNet):

	preds = []
	locs = []
	pixels = frame
	# detect faces in the image
	faces = detector.detect_faces(pixels)
	faceimgs=[]
	for face in faces:
		b = face['box']
		bt = (b[0], b[1], b[0]+b[2], b[1]+b[3])
		print(face)

		confidence = face['confidence']

		if confidence > 0.3:
			"""
			# inside frame
			(bt[0], bt[1]) = (max(0, bt[0]), max(0, bt[1]))
			(bt[2], bt[3]) = (min(b[2] - 1, bt[2]), min(b[3] - 1, bt[3]))
			"""
			# faceimg = frame[startY:endY, startX:endX]
			faceimg = frame[bt[1]-20:bt[3]+20, bt[0]-20:bt[2]+20]
			if faceimg.any():
				faceimg = cv2.cvtColor(faceimg, cv2.COLOR_BGR2RGB)
				faceimg = cv2.resize(faceimg, (224, 224))
				faceimg = img_to_array(faceimg)
				faceimg = preprocess_input(faceimg)
				faceimgs.append(faceimg)
			locs.append(bt)

	# only make a predistartYctions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faceimgs = np.array(faceimgs, dtype="float32")
		preds = maskNet.predict(faceimgs, batch_size=4)

	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-i", "--input", type=str, default='input/i1.mp4',
	help="input file path")
ap.add_argument("-o", "--output", type=str, default='output/o1.mp4',
	help="output file path")
ap.add_argument("-v", "--csv", type=str, default='output/o1.csv',
	help="output file path")
args = vars(ap.parse_args())

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs=cv2.VideoCapture(args["input"])
time.sleep(0.5)

frame_width=720
frame_height = int(vs.get(4)*720/vs.get(3))
frame_size = (frame_width,frame_height)
fps = int(vs.get(5))

# Create output object
output = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

# CSV
csvfile = open(args["csv"], 'w')
csvwriter = csv.writer(csvfile)

i=1
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 720 pixels
	ret, frame = vs.read()
	if (ret != True):
		break

	frame = imutils.resize(frame, width=720)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, maskNet)
	# loop over the detected face locations and their corresponding
	# locations
	noMasked = 0
	maskedLocs = []
	ID = 1
	if len(locs) > 1:
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "No Mask"
			if mask > withoutMask:
				label = "Mask"
				noMasked += 1
				maskedLocs.append(box)
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "ID-{}:{}".format(ID, label)
			ID+=1

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


	row = [i, len(locs), noMasked, locs, maskedLocs]
	csvwriter.writerow(row)
	# show the output frame
	output.write(frame)
	#cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	print("wrote", i)
	i+=1
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
output.release()
