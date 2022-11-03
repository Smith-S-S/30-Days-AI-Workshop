import cv2
import imutils
img= cv2.imread("agarwals-images-jagadamba-centre-visakhapatnam-fashion-photographers-yn7k0.jpg.webp")
resize= imutils.resize(img,width=600)
cv2.imshow("sm",resize)
cv2.imshow("sm2",img)
#cv2.waitKey(0)


"""# smoothing the image
gaussiam_blur = cv2.GaussianBlur(img,(21,21),0)
cv2.imshow("image",img)
cv2.waitKey(0)

cv2.imshow("gaussiam_blur",gaussiam_blur)
cv2.waitKey(0)

#put the text

cv2.rectangle()
cv2.putText("")"""


# moving object detection
import time
cam=cv2.VideoCapture(0)
time.sleep(1) #1 sec of dely
first_frame= None
area =500
while True:
    _,img=cam.read()
    text= "Normal"
    img=imutils.resize(img,width=500)
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussiam_blur = cv2.GaussianBlur(img,(21,21),0)

    if first_frame is None:
        first_frame=gaussiam_blur
        continue

    imgDiff= cv2.absdiff(first_frame,gaussiam_blur)
    thresh=cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)
    thresh_img= cv2.dilate(thresh ,None, iterations=2)
    cnt= cv2.findCountours(thresh_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt=imutils.grab_count(cnt)
    for c in cv2:
        if cv2.contourArea(c) < area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x + w, y + h), (0, 255, 0))
        text= "Moving object dected"
    print(text)
    cv2.putText(img, text,(10,20),CV2. FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)(0, 0, 255),2)
    cv2.imshow("cameraFeed", img)
    key=cv2.waitKey(1)& OxFF

    if key==ord("q"):
        break


