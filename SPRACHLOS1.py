import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import random

# Klasse für die Vollbildanwendung
class FullScreenApp:
    def __init__(self, master):
        self.master = master
        self.master.attributes('-fullscreen', True)  # Setze Vollbild-Attribut für das Fenster
        self.start_screen()

    def start_screen(self):
        image = Image.open("./graphics/sprachlos.png")  # Öffne das Bild
        screen_width = self.master.winfo_screenwidth()  # Breite des Bildschirms
        screen_height = self.master.winfo_screenheight()  # Höhe des Bildschirms
        image = image.resize((screen_width, screen_height), Image.LANCZOS)  # Ändere die Größe des Bildes

        self.background_image = ImageTk.PhotoImage(image)  # Erzeuge ein PhotoImage-Objekt

        self.background_label = tk.Label(self.master, image=self.background_image)  # Erzeuge ein Label-Widget
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Positioniere das Label im Fenster

        self.master.after(3000, self.destroy_start_screen)  # Nach 3000 Millisekunden das Startbildschirm zerstören

    def destroy_start_screen(self):
        self.background_label.destroy()  # Zerstöre das Label-Widget
        self.main_game()  # Starte das Hauptspiel

    def main_game(self):
        self.cap = cv2.VideoCapture(0)  # Öffne die Videoaufnahme
        self.model_dict = pickle.load(open('./model.p', 'rb'))  # Lade das Modell aus der Datei
        self.model = self.model_dict['model']  # Das Modell
        self.mp_hands = mp.solutions.hands  # Handerkennung von Mediapipe
        self.mp_drawing = mp.solutions.drawing_utils  # Hilfsfunktionen zum Zeichnen von Mediapipe
        self.mp_drawing_styles = mp.solutions.drawing_styles  # Zeichenstile von Mediapipe
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # Initialisiere die Handerkennung
        self.labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
            10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
            19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'SCH'
        }  # Wörterbuch für die Zuordnung der Labels
        self.color = (0, 150, 244)  # Farbe für die Zeichnung

        self.frame = tk.Frame(self.master)  # Erzeuge ein Frame-Widget
        self.frame.pack(fill="both", expand=True)  # Fülle den Raum im Fenster

        self.video_label = tk.Label(self.frame)  # Erzeuge ein Label-Widget
        self.video_label.pack(expand=True)  # Fülle den Raum im Frame

        self.info_frame = tk.Frame(self.frame)  # Erzeuge ein Frame-Widget für Informationen
        self.info_frame.pack(expand=True)  # Zeige das Info-Frame an, expand=True -> Raum gleichmäßig aufteilen

        self.is_playing = False  # Variable für den Spielstatus
        self.score = 0  # Spielstand
        self.current_character = None  # Aktuelles Zeichen
        self.remaining_time = 60 # Zeit

        self.character_label = tk.Label(self.info_frame, text=f'Bereit? Klicke auf Start!', font=(None, 25))  # Label für das Zeichen
        self.character_label.grid(row=0)

        self.score_label = tk.Label(self.info_frame, text=f'Score: {self.score}')  # Label für den Spielstand
        self.score_label.grid(row=1)  # Zeige den Spielstand an

        self.timer_label = tk.Label(self.info_frame, text=f'Time Left: {self.remaining_time}') # Label für die Zeit
        self.timer_label.grid(row=2)

        self.start_button = tk.Button(self.info_frame, text="Start Game", fg="#F49600", command=self.start_game)  # Start-Button
        self.start_button.grid(row=3)  # Zeige den Start-Button an

        self.update_frame()  # Aktualisiere das Videobild

    def generate_new_character(self):
        self.current_character = random.choice(list(self.labels_dict.values()))  # Zufälliges Zeichen auswählen
        self.character_label.config(text=f'Buchstabe: {self.current_character}', font=(None, 25)) # Schreibe den neuen Buchstaben in das Label

    def check_prediction(self, predicted_character):
        if self.is_playing and predicted_character == self.current_character:  # Überprüfe die Vorhersage
            self.score += 1  # Erhöhe den Spielstand
            self.score_label.config(text=f'Score: {self.score}')  # Aktualisiere das Spielstand-Label
            self.generate_new_character()  # Generiere ein neues Zeichen

    def start_game(self):
        self.is_playing = True
        self.remaining_time = 60 # Zeitstand
        self.score = 0  # Spielstand
        self.timer_label.config(text=f'Time Left: {self.remaining_time}') # Schreibe die Zeit in das Label
        self.score_label.config(text=f'Score: {self.score}') # Schreibe den Score in das Label
        self.generate_new_character()
        self.update_timer()

    def update_timer(self):
        if self.remaining_time > 0:
            self.remaining_time -= 1
            self.timer_label.config(text=f'Time Left: {self.remaining_time}')
            self.master.after(1000, self.update_timer)
        else:
            self.end_game()

    def end_game(self):
        self.is_playing = False
        self.start_button.config(state="normal") # Button-Reset
        self.character_label.destroy()  # Zerstöre das Zeichen-Label, sodass nur das neue dasteht


    def update_frame(self):
        ret, frame = self.cap.read()  # Lese ein Frame von der Videoaufnahme
        if not ret:
            return
        frame = cv2.flip(frame, 1)  # Spiegele das Frame horizontal

        data_aux = []  # Hilfsliste für die Daten
        x_ = []  # Liste für die x-Koordinaten
        y_ = []  # Liste für die y-Koordinaten
        H, W, _ = frame.shape  # Höhe und Breite des Frames
        results = self.hands.process(frame)  # Verarbeite das Frame mit der Handerkennung

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Erkenne die Hand
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=self.color, thickness=2, circle_radius=4),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=self.color, thickness=2, circle_radius=2)
            )  # Zeichne die Handpunkte

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
            prediction = self.model.predict([np.asarray(data_aux)])  # Mache eine Vorhersage
            predicted_character = self.labels_dict[int(prediction[0])]  # Vorhergesagtes Zeichen
            self.check_prediction(predicted_character)  # Überprüfe die Vorhersage
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 4)  # Zeichne ein Rechteck um die Hand
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, self.color, 3,
                        cv2.LINE_AA)  # Zeichne das vorhergesagte Zeichen

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Konvertiere das Frame in das richtige Farbformat
        image = Image.fromarray(frame)  # Erzeuge ein PIL-Bild aus dem Frame
        image = ImageTk.PhotoImage(image)  # Erzeuge ein PhotoImage-Objekt aus dem PIL-Bild

        if not hasattr(self, 'video_label'):
            self.video_label = tk.Label(self.master)
            self.video_label.pack(fill="both", expand="yes")  # Fülle den Raum im Fenster
        self.video_label.config(image=image)  # Zeige das Video-Label mit dem aktuellen Frame an
        self.video_label.image = image  # Aktualisiere das Image-Attribut des Labels

        self.master.after(5, self.update_frame)  # Aktualisiere das Frame nach 5 Millisekunden


root = tk.Tk()  # Erzeuge ein Tkinter-Fenster
app = FullScreenApp(root)  # Erzeuge eine Vollbildanwendung
root.bind('<Escape>', lambda event: root.attributes('-fullscreen', False))  # ESC-Taste zum Verlassen des Vollbildmodus
root.bind('<Command-q>', lambda event: root.destroy())  # Tastenkombination zum Beenden des Programms (macOS)
root.bind('<Alt-F4>', lambda event: root.destroy())  # Tastenkombination zum Beenden des Programms (Windows)
root.mainloop()  # Starte die Tkinter-Schleife
