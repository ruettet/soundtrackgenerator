import numpy as np
import cv2

cap = cv2.VideoCapture(0)

yellow_threshold = 1000000

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([20, 100, 100])
    upper_blue = np.array([30, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    count = mask.sum()

    if count > yellow_threshold:
        print('I can see yellow')
    else:
        print('I see no yellow')

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()