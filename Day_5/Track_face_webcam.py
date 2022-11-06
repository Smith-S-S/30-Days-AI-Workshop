import cv2
train = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

vd = cv2.VideoCapture(0)

while True:
    success, frame = vd.read()
    if success== True:
        gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        com = train.detectMultiScale(gr)
        #print (com)
        #print (frame)
        for x, y, w, h in com:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.imshow("s", frame)
        key = cv2.waitKey(1)
        if key == 32:
            break
    else:
        print ("completed:")
        break
