import cv2
#trainedDataset= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

trainedDataset=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# x=haarcascade_frontalface_default
# av=cv2.CascadeClassifier(x)

#pip install opencv-python

# read an image
img= cv2.imread('images/pp.jpg')


# showing the image
#cv2.imshow('smith',img)
#cv2.waitKey() #wait to show the image


# gray scale image

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#cv2.imshow('GRAY',gray)
"""cv2.imshow("sm",gray)
cv2.waitKey()"""

#FINDING OUR FACE
faces= trainedDataset.detectMultiScale(gray)
print("faces=  ", faces)

# square in photo

for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+h,y+w),(255,0,0),10)
    # rgb colours 0 to 255 {255,0,0}, thickness(2)
cv2.imshow('smith', img)
cv2.waitKey()

#cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0)
