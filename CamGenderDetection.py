import cv2
import dlib
import numpy as np
import pafy

import GenderPredictor as gc

class_names = ['male','female']


print (cv2.getVersionString())

face_predictor_path = 'shape_predictor_5_face_landmarks.dat'
gender_predictor_path = 'model_90000_96.dat'

gpredictor = gc.GenderPredictor(gender_predictor_path)

# init dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_predictor_path)

#init gender detection
gpredictor.build_model()

#start video streaming
#url = 'https://youtu.be/9cmWFdBYM0M'
#url= 'https://youtu.be/vGe1Eb_dyh4'
#vPafy = pafy.new(url)
#print (vPafy.streams)
#play = vPafy.getbest(preftype="mp4")
#play = vPafy.streams[1]
#print (play)

#start video capture
#cap = cv2.VideoCapture(play.url)
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cat_name = ("M", "F")

while(True):
    #capture video image
    ret, frame = cap.read()

    dets = detector(frame, 0)
    capture = 0
    if len(dets) > 0:
        #

        for detection in dets:
            color = (0,0,255)
            faces = dlib.full_object_detections()
            faces.append(predictor(frame, detection))
            images = dlib.get_face_chips(frame, faces, size=64, padding=0.40)  # extract faces images

            rimg = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)  # roll to rgb. the model need rgb images

            cv2.imshow('face' + str(capture), images[0])  # display face image

            capture += 1
            prediction = gpredictor.predict(rimg)
            if (len(prediction) > 0):
                # print (prediction)
                c = np.argmax(prediction)
                title = cat_name[c]
                color = (255, 0, 0) if c == 0 else (128, 0, 255)

                if abs(prediction[0][0] - prediction[0][1]) < 0.5:
                    color = (0, 255, 0)
                else:
                    cv2.putText(frame, title, (detection.tl_corner().x, detection.tl_corner().y),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
                cv2.rectangle(frame, (detection.tl_corner().x, detection.tl_corner().y),
                              (detection.br_corner().x, detection.br_corner().y), color)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27 :
            break


cap.release()
cv2.destroyAllWindows()
