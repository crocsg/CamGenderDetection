import cv2
import dlib
import numpy as np

import GenderPredictor as gc

class_names = ['male','female']


print (cv2.getVersionString())

face_predictor_path = 'shape_predictor_5_face_landmarks.dat'
gender_predictor_path = 'model_23000_92.dat'

gpredictor = gc.GenderPredictor(gender_predictor_path)

# init dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_predictor_path)

#init gender detection
gpredictor.build_model()

#start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cat_name = ("male", "female")

while(True):
    #capture video image
    ret, frame = cap.read()

    dets = detector(frame, 0)
    print (dets)
    if len(dets) > 0:
        faces = dlib.full_object_detections()

        for detection in dets:
            faces.append(predictor(frame, detection))

        images = dlib.get_face_chips(frame, faces, size=64, padding=0.40)   # extract faces images

        for img in images:
            rimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # roll to rgb. the model need rgb images

            cv2.imshow('face', img) # display face image

            prediction = gpredictor.predict (rimg)
            if (len(prediction) > 0):
                #print (prediction)
                c = np.argmax(prediction)
                title = cat_name[c] + " " + str(prediction[0])

                if abs(prediction[0][0] - prediction[0][1]) > 0.5:
                    cv2.putText(frame, title, (10,40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0))


    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27 :
            break


cap.release()
cv2.destroyAllWindows()