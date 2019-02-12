import cv2
import numpy as np



detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer\\trainingData.yml")
id=0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 5, 0, 1, 4)

while(True):
    ret,img=cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if (id == 1):
               id="Rohan"
        if (id == 2):
                id="MS Dhoni"
                
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h),font, 255)
    cv2.imshow('face_rohan',img);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()
    
