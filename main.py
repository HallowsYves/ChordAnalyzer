import cv2

videoCapture = cv2.VideoCapture(0)

while True:
    valid, frame = videoCapture.read()
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
videoCapture.release()
cv2.destroyAllWindows()