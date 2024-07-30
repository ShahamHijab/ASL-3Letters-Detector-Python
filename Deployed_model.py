import cv2
import mediapipe as mp
import pickle
import numpy as np
import streamlit as st # for deployment

def load():
    #load model
    model_dic = pickle.load(open('./model.p', 'rb'))
    return model_dic['model']

def process(frame, cap, model, labels_dic, mp_hands, mp_drawing, mp_drawing_styles, hands):
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks( # if hand detected, landmarks will be drawn in camera frame
                frame,  # image to draw
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
        if len(data_aux) == 42:
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            # prediction
            prediction = model.predict([np.asarray(data_aux)])
             # for displaying predicted letter in frame
            predicted_char = labels_dic[int(prediction[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    return frame

def app():
    model = load()
    if model is None:
        exit()

    labels_dic = {0: 'A', 1: 'B', 2: 'L'}

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    # For navbar
    st.markdown(
        """
        <style>
            .navbar {
                background-color: #A08FFF;
                color: white;
                padding: 10px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .navbar .logo {
                font-size: 28px;
                font-weight: bold;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    #To create a navbar
    st.markdown(
        """
        <div class="navbar">
            <div class="logo">Sign Language Detector - AI Project</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Using Markdown syntax to control font size and line spacing
    st.markdown("<h3 style='font-size: 18px; margin-bottom: 1px;'>Name1</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 18px; margin-bottom: 1px;'>Name2</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 18px; margin-bottom: 1px;'>Name3</h3>", unsafe_allow_html=True)

    # Names were for project creators
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image("signlanguage2.png", width=300)

    run_app = st.checkbox("Use Camera") # permission for using camera

    with col2:
        if run_app:   # if camera allowed
            cap = cv2.VideoCapture(0) #use camera
            if not cap.isOpened():
                st.error("Failed to open camera.")
                return

            frame_placeholder = st.empty()

            while run_app:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame.")
                    break

                processed_frame = process(frame, cap, model, labels_dic, mp_hands, mp_drawing, mp_drawing_styles, hands) # for predictions
                frame_placeholder.image(processed_frame, channels="BGR")

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app()
