#in this method we take list operation to take image from image files
import numpy
import cv2
import face_recognition
import os

path = "image_attendance"
images = []
class_name = []

my_list = os.listdir(path)
print(my_list)
print (path)

for i in my_list:
    current_inage =cv2.imread(f"{path}/{i}")
    images.append(current_inage)
    class_name.append(os.path.splitext(i)[0])
print (class_name)

#function to create the face_encodings

def Find_encding(images):
    encode_list=[] # all the encodings
    for j in images:
        j = cv2.cvtColor(j, cv2.COLOR_BGR2RGB)
        face_encode = face_recognition.face_encodings(j)[0]
        encode_list.append(face_encode)
    return encode_list
# First you have the digitized picture

encodings_list = Find_encding(images)
#print (len(encodings_list[0]))
print ("Encoding Complete")

# to find the matches in our encodings with the web camp
cam= cv2.VideoCapture(0)
while True:
    success, img= cam.read()
    image_smat = cv2.resize(img,(0,0),None,0.25)


img_Elon = face_recognition.load_image_file("images/elon.jpg")
img_Elon= cv2.cvtColor(img_Elon, cv2.COLOR_BGR2RGB)


# now going to find the faces and the encodes

face_locate= face_recognition.face_locations(img_Elon)[0] # it give as 4 values
face_encode= face_recognition.face_encodings(img_Elon)

# draw the rectangle

cv2.rectangle(img_Elon,(face_locate[3],face_locate[0]),(face_locate[1],face_locate[2]),(255,8,23),2)



#cv2.imshow("elon",img_Elon)
#cv2.waitKey(0)
