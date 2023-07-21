import os
import pickle

import mediapipe as mp
import cv2

#MediaPipe für die Handerkennung
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

#Durchlaufe jedes Verzeichnis im Datenverzeichnis
for dir_ in os.listdir(DATA_DIR):
    if dir_ == '.DS_Store':
        continue
    
    # Durchlaufe jede Bild
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        
        data_aux = []

        # Listen für die x- und y-Koordinaten
        x_ = []
        y_ = []

        #Bild Laden und in RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        
        #Falls Handmarker erkannt wurden
        if results.multi_hand_landmarks:
            
            #Durchlaufe jeden Punkt
            for hand_landmarks in results.multi_hand_landmarks:
                
                #X und Y sammeln
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

            #Füge die Daten und das Label zu den Listen hinzu
            data.append(data_aux)
            labels.append(dir_)

#Speichere die Daten und Labels in einer Pickle-Datei
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
