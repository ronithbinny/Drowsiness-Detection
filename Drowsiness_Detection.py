import cv2
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
ti = 0

while True :
    
    _,img = cap.read()
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = detector(grey)
    
    for face in faces :
        x , y = face.left(), face.top()
        x1 , y1 = face.right(), face.bottom()
        
        cv2.rectangle(img, (x,y), (x1,y1), (0,255,0), 2)
        
        landmarks = predictor(grey,face)
        
        box1 = (landmarks.part(37).x,landmarks.part(37).y,landmarks.part(40).x,landmarks.part(40).y)
        box2 = (landmarks.part(43).x,landmarks.part(43).y,landmarks.part(46).x,landmarks.part(46).y)
        
        cv2.rectangle(img, (landmarks.part(37).x,landmarks.part(37).y), 
                      (landmarks.part(40).x,landmarks.part(40).y),(0,255,0),1)
        cv2.rectangle(img, (landmarks.part(43).x,landmarks.part(43).y), 
                      (landmarks.part(46).x,landmarks.part(46).y),(0,255,0),1)
        
        distance1 = int(landmarks.part(40).y) - int(landmarks.part(37).y)
        distance2 = int(landmarks.part(46).y) - int(landmarks.part(43).y)
        print(distance1,distance2)
        ti = ti + 1
        
        if distance1 <= 5 and distance2 <= 5 :
            print("SLEEPING")
        
    cv2.imshow('img',img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# a-left eye, b- right eye
