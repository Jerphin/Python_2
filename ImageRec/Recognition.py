import os
import cv2
import face_recognition
import mediapipe as mp
import numpy as np

#Load known faces
known_face_encodings=[]
known_face_names=[]

for filename in os.listdir('Faces'):
    path=os.path.join('Faces',filename)
    image=face_recognition.load_image_file(path)
    encodings=face_recognition.face_encodings(image)
    if encodings:
        known_face_encodings.append(np.array(encodings[0]))
        name=os.path.splitext(filename)[0]
        known_face_names.append(name)

#Mediapipe face detection setup
mp_face_detection=mp.solutions.face_detection
mp_drawing=mp.solutions.drawing_utils
cap=cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.6)as face_detection:
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        face_locations=face_recognition.face_locations(rgb_frame)
        face_encodings=face_recognition.face_encodings(rgb_frame,face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            name="Unknown"
            #Use the shortest distance for best match
            face_distances=face_recognition.face_distance(known_face_encodings,face_encoding)
            if len(face_distances)>0:
                best_match_index=face_distances.argmin()
                if matches[best_match_index]:
                    name=known_face_names[best_match_index]
            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
            cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.imshow("Face Recognition",frame)

        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




#how to run this code
# in powershell
# PS C:\Users\jerph> cd D:\ImageRec
# PS D:\ImageRec> C:\Users\jerph\dlib_env\Scripts\Activate.ps1
# (dlib_env) PS D:\ImageRec> python Recognition.py"""