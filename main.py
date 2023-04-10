import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:

    _, frame = cap.read()
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)

    white_lower = np.array([0, 0, 0], np.uint8)
    white_upper = np.array([0, 0, 255], np.uint8)

    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

    kernel = np.ones((5,5), "uint8")

    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

    white_mask = cv2.dilate(white_mask, kernel)
    res_white = cv2.bitwise_and(frame, frame, mask=white_mask)

    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours) :
        area = cv2.contourArea(contour)
        if (area > 300) :
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(frame, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours) :
        area = cv2.contourArea(contour)
        if (area > 300) :
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, "White", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    cv2.imshow("detectionRed",frame)

    if cv2.waitKey(100) & 0xff == ord('q') :
        cap.release()
        cv2.destroyAllWindows()
        break

