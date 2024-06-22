#DATA SET
import os
import mediapipe as mp #for vision task .here use for hand tracking
import cv2       # for computer vision
import matplotlib.pyplot as plt
import pickle

mp_hands=mp.solutions.hands     # initializing hand module
mp_drawing=mp.solutions.drawing_utils  # for drawing landmarks
mp_drawing_styles=mp.solutions.drawing_styles
hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3) # detection for detecting hands

data=[]
label=[]


data_dir = r'D:\Torture\Labs\4th sem labs\Intro to AI\Projects\Final\Extras\lastTest\Project-1\data' # for data


for dir_ in os.listdir(data_dir):
        # taking pictures from directory

            for img_path in os.listdir(os.path.join(data_dir, dir_)):
                data_aux=[]
                img = cv2.imread(os.path.join(data_dir, dir_, img_path))

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting to RGB
                result=hands.process(img_rgb)
                if result.multi_hand_landmarks: # if hand detected
                        for hand_landmarks in  result.multi_hand_landmarks:
                                for i in range(len(hand_landmarks.landmark)):
                                        #extracting coordinates of landmark
                                        x=hand_landmarks.landmark[i].x
                                        y=hand_landmarks.landmark[i].y
                                        # storing coordinates
                                        data_aux.append(x)
                                        data_aux.append(y)

                        data.append(data_aux)
                        label.append(dir_)
f=open('data.pickle','wb')
pickle.dump({'data':data,'label':label},f)  # storing data and labels to file

f.close()