import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading
import random
import time

class FullScreenApp:
    def __init__(self, master):
        self.master = master
        self.master.attributes('-fullscreen', True) # Vollbildmodus 
        self.is_playing = False  
        self.start_screen()

    def start_screen(self):
        image = Image.open("./graphics/sprachlos.png")
        
        #Bildgröße anpassen
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        image = image.resize((screen_width, screen_height), Image.LANCZOS)

        #In Tkinter PhotoImage umwandeln
        self.background_image = ImageTk.PhotoImage(image)

        #Label
        self.background_label = tk.Label(self.master, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1,)

        self.master.after(5000, self.destroy_start_screen)

    def destroy_start_screen(self):
        self.background_label.destroy()
        self.main_menu()

    def main_menu(self):
        #Mediapipe Handtracking
        self.cap = cv2.VideoCapture(0)
        self.model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = self.model_dict['model']
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.labels_dict = {
                0: 'A', 
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
        self.color = (0, 150, 244)  
        
        #Vollbild Frame
        self.frame = tk.Frame(self.master)
        self.frame.pack(fill="both", expand="yes")

        #Video einfügen
        self.video_label = tk.Label(self.frame)
        self.video_label.pack(fill="both", expand="yes")

        #Start Button erstellen
        self.start_button = tk.Button(self.frame, text="Spiel starten", command=self.start_game)
        self.start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.update_frame()

    def start_game(self):
        self.start_button.destroy()
        self.is_playing = True
        self.score = 0
        self.main_game()
        
    def main_game(self):
        self.current_score = 0
        self.current_character = None

        # Score 
        self.score_label = tk.Label(self.frame, text=f'Score: {self.current_score}')
        self.score_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        self.generate_new_character()

    def generate_new_character(self):
        #Zeichen auswählen 
        self.current_character = random.choice(list(self.labels_dict.values()))
        self.character_label = tk.Label(self.frame, text=f'Zeichen: {self.current_character}')
        self.character_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
            
    def check_prediction(self, predicted_character):
        if self.is_playing and predicted_character == self.current_character:
            #Input überpfrüen
            self.current_score += 1
            self.score_label.config(text=f'Punktzahl: {self.current_score}')
            self.character_label.destroy()
            self.generate_new_character()

    def update_frame(self):
        # Lese ein Bild von der Kamera ein
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
            
        data_aux = []
        x_ = []
        y_ = []
        H, W, _ = frame.shape
        results = self.hands.process(frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=self.color, thickness=2, circle_radius=4),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=self.color, thickness=2, circle_radius=2)
            )

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
            prediction = self.model.predict([np.asarray(data_aux)])
            predicted_character = self.labels_dict[int(prediction[0])]
            self.check_prediction(predicted_character)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, self.color, 3, cv2.LINE_AA)
            
        #Bildschirmgröße abrufen
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
            
        #Bildgröße anpassen an Bildschirm anpassen
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (screen_width, screen_height))
        image = Image.fromarray(frame)  
        image = ImageTk.PhotoImage(image)
            
        if not hasattr(self, 'video_label'):
            self.video_label = tk.Label(self.video_frame)
            self.video_label.pack(fill="both", expand="yes")    
        #Video Label Bild aktualisieren
        self.video_label.config(image=image)
        self.video_label.image = image

        self.master.after(5, self.update_frame)


root = tk.Tk()
app = FullScreenApp(root)
root.bind('<Escape>', lambda event: root.attributes('-fullscreen', False))
root.bind('<Command-q>', lambda event: root.destroy())
root.bind('<Alt-F4>', lambda event: root.destroy())

root.mainloop()
