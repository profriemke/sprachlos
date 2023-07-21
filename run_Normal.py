import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

labels_dict = {0: 'A', 
               1: 'B', 
               2: 'C', 
               3: 'D', 
               4: 'E', 
               5: 'F', 
               6: 'G', 
               7: 'H', 
               8: 'I', 
               9: 'K', 
               10: 'L', 
               11: 'M', 
               12: 'N', 
               13: 'O', 
               14: 'P', 
               15: 'Q', 
               16: 'R', 
               17: 'S', 
               18: 'T', 
               19: 'U', 
               20: 'V', 
               21: 'W', 
               22: 'X', 
               23: 'Y',
               24: 'SCH'}

# Farbe
color = (0, 150, 244)

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Erste Hand die erkannt wird
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Specify the color for the landmarks
        mp_drawing.draw_landmarks(
            frame, # image to draw
            hand_landmarks, # model output
            mp_hands.HAND_CONNECTIONS, # hand connections
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2))
        
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
