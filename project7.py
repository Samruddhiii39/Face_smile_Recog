import cv2
rec = cv2.CascadeClassifier("C:/Python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
rec2 = cv2.CascadeClassifier("C:/Python/Lib/site-packages/cv2/data/SMILE.xml")
cam = cv2.VideoCapture(0)
while True :
    ret , video = cam.read()
    color = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    face = rec.detectMultiScale(
        color,
        scaleFactor=2,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in face:
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,0,1200),2)

    smile = rec2.detectMultiScale(
        color,
        scaleFactor=1.8,
        minNeighbors=5,
        minSize= (30,30)
    )
    for (x,y,w,h) in smile:
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow("live_vid", video)
    if cv2.waitKey(15) == ord("0"):
        break
cam.release()
