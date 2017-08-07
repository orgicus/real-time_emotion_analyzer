import cv2
import sys
import json
import time
import numpy as np
from keras.models import model_from_json


emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
cascPath = sys.argv[1]

faceCascade = cv2.CascadeClassifier(cascPath)

# load json and create model arch
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('model.h5')

def display_emotion(frame,label,value,x,y,bw):
    bx = 80
    cv2.putText(frame,label,(x,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)
    cv2.rectangle(frame, (bx, y), (int(bx+bw*value), y+10), (0, 255, 0), 2)
    cv2.putText(frame,str(value),(bx+bw+75,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)

def predict_emotion(face_image_gray): # a single cropped face
    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    # cv2.imwrite(str(index)+'.png', resized_img)
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]

video_capture = cv2.VideoCapture(0)
# gain speed: reduce camera resolution
video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,320)
video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,240)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY,1)


    faces = faceCascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    emotions = []
    # Draw a rectangle around the faces
    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        face_image_gray = img_gray[y:y+h, x:x+w]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        angry, fear, happy, sad, surprise, neutral = predict_emotion(face_image_gray)

        display_emotion(frame,"angry",angry,10,5,100)
        display_emotion(frame,"fear",fear,10,20,100)
        display_emotion(frame,"happy",happy,10,35,100)
        display_emotion(frame,"sad",sad,10,50,100)
        display_emotion(frame,"surprise",surprise,10,65,100)
        display_emotion(frame,"neutral",neutral,10,80,100)

        # print "angry",angry, "fear",fear, "happy",happy, "sad",sad, "surprise",surprise, "neutral",neutral
        # with open('emotion.txt', 'a') as f:
        #     f.write('{},{},{},{},{},{},{}\n'.format(time.time(), angry, fear, happy, sad, surprise, neutral))


    # Display the resulting frame
    cv2.imshow('Video', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
