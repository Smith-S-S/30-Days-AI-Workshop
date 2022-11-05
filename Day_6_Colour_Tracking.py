import cv2
import imutils
redLower=(15, 164, 162)
redUpper = (45, 232, 255)

camera =cv2.VideoCapture(0)

while True:
    (grabbed, frame)= camera.read()
    frame= imutils.resize(frame,width=600) #rezising
    blurred=cv2.GaussianBlur(frame, (11, 11), 0) #smoothining the image
    hsv= cv2.cvtColor (blurred, cv2.COLOR_BGR2HSV) # convert to hsv
    #gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask=cv2.inRange(hsv, redLower, redUpper) # mask green
    mask=cv2.erode (mask, None, iterations=2)
    mask=cv2.dilate (mask, None, iterations=2)
    # to enhance the image

    cnts=cv2. findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) [-2]
    # finding the neighbourhood
    # finding the counters of the complete frame
    center= None
    if len(cnts) > 0:
        # going to draw the small red colour circle for that we need center
        c= max(cnts, key=cv2.contourArea)
        ((x, y), radius)= cv2.minEnclosingCircle(c)
        # x,y the yellow circle point and radius of it
        M= cv2.moments(c)
        center=(int(M["m10"] / M["m00"]), int(M["m01"]/M["m00"]))
        if radius > 10:
            cv2.circle(frame,(int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame,center, 5, (0, 0, 255),-1)
            if radius > 250:
                print("stop")
            else:
                if (center[0] < 150):
                    print(("Left"))
                elif (center[0] > 450):
                    print("Right")
                elif (radius < 250):
                    print("Front")
                else:
                    print("Stop")

    #cv2.imshow("Frame",frame)
    cv2.imshow("s", frame)
    key = cv2.waitKey(5)
    if key == 32:
        break
















