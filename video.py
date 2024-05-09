import cv2
import numpy as np
import os
import pathlib
import tensorflow as tf
from imutils.object_detection import non_max_suppression

camera = cv2.VideoCapture(0)

emotion_model = tf.keras.models.load_model('backend/models/model_lessparameter,55E.h5')

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
face_model = cv2.CascadeClassifier(str(cascade_path))

categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class Capture:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.path = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
        self.face_model = face_model
        self.emotion_model = emotion_model
   
    def show_video(self, save=None):
        ret, frame = self.video.read()
        frame = cv2.resize(frame, (850, 480))
        frame = cv2.flip(frame, 180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        faces = non_max_suppression(faces)
    
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            x_min, y_min, x_max, y_max = x, y, x + width, y + height
            emotion, probabilities = self.predict(frame[y_min:y_max, x_min:x_max])
            self.draw_emotion(frame, emotion, probabilities, (x, y), (width, height))
            
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
    
    def predict(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desired_size = (48, 48)
        resized_img = cv2.resize(gray_image, desired_size)
        resized_img = np.expand_dims(resized_img, axis=-1)
        prediction = self.emotion_model.predict(resized_img.reshape(1, 48, 48, 1))
        final_prediction = np.argmax(prediction)
        emotion = categories[final_prediction].upper()
        probabilities = prediction[0]
        return emotion, probabilities
    
    def draw_emotion(self, frame, emotion, probabilities, position, size):
        font_scale = 1
        font_thickness = 2
        font_color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Display the detected emotion
        text = f"Detected Emotion: {emotion.upper()}"
        text_width, text_height = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = position[0] + (size[0] - text_width) // 2
        text_y = position[1] + size[1] + 30
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

        # Display the probability of the detected emotion if it's not 'neutral'
        if emotion.lower() != 'neutral':
            detected_prob = probabilities[categories.index(emotion.lower())]
            prob_percent = "{:.2f}%".format(detected_prob * 100)
            prob_text = f"Probability: {prob_percent}"
            prob_y = text_y + 30
            cv2.putText(frame, prob_text, (text_x, prob_y), font, font_scale, font_color, font_thickness)

# Define the generate_video function
def generate_video(camera, save=None):
    while True:
        frame = camera.show_video(save=save)
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


