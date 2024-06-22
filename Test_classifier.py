# Test classifier

import cv2
import mediapipe as mp
import pickle
import numpy as np


model_dic=pickle.load(open('./model.p','rb')) #load model
model=model_dic['model']

cap=cv2.VideoCapture(0) # default camera for video capturing

# for hands landmarks
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)

labels_dic={0:'A',1:'B',2:'L'} # labels on basis of their classes

while True:
    data_aux=[]
    x_=[]
    y_=[]

    ret,frame=cap.read()

    H,W,_=frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result=hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in  result.multi_hand_landmarks:
                                mp_drawing.draw_landmarks( # if hand detected, landmarks will be drawn in camera frame
                                        frame,      #image to draw
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS, #hand connections
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style(),
                                )
        for hand_landmarks in  result.multi_hand_landmarks:
                                for i in range(len(hand_landmarks.landmark)):
                                        x=hand_landmarks.landmark[i].x
                                        y=hand_landmarks.landmark[i].y
                                        data_aux.append(x)
                                        data_aux.append(y)
                                        x_.append(x)
                                        y_.append(y)
        print("LEEEEEEE",len(data_aux))
        if len(data_aux) == 42:
                x1=int(min(x_)*W)
                y1=int(min(y_)*H)
                x2=int(max(x_)*W)
                y2=int(max(y_)*H)
                # prediction
                prediction=model.predict([np.asarray(data_aux)])
                  # for displaying predicted letter in frame
                predicted_char=labels_dic[int(prediction[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)