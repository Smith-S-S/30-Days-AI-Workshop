import numpy as np
import imutils
import time
import cv2


prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confThresh=0.2

# initialize the list of class labels MobileNet SSD was trained to
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# picking random colour for each class

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt,model)

# initialize the video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)


# loop over the frames from the video stream
while True:

    # grab the frame
    # to have a maximum width of 400 pixels
	_,frame = vs.read()
	frame = imutils.resize(frame, width=500)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2] # to get the height and weight


	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5) #prebuit model frame is 300x300
	#conveting to blop image

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob) # make as an input
	detections = net.forward() # geat all the info about the image
    #detShape = detections.shape[2]

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		#print(i)
		confidence = detections[0, 0, i, 2] # this give the confidence

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence>confThresh:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
			idx = int(detections[0,0,i,1]) # it give the index number


			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int") # it give the bounding box info(3:7[3,4,5,6])

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)# get the name from CLASS
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			
			y = startY - 15 if startY - 15 > 15 else startY + 15



			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()





