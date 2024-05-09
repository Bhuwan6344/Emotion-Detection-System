import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, Response, request
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
import random
import tensorflow as tf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Define directories for images and models
IMAGE_DIRS = ['/Users/sam/Downloads/Dataset/train', '/Users/sam/Downloads/Dataset/test']
MODEL_FILES = ['/Users/sam/Desktop/7th sem/web-app/backend/models/model_lessparameter,55E.h5',
               '/Users/sam/Desktop/7th sem/web-app/backend/models/model_lessparameter,55E.h5',
               '/Users/sam/Desktop/7th sem/web-app/backend/models/model_lessparameter,55E.h5']
CSV_FILES = ['static/csv/history_lessparameter,55E.csv', 'static/csv/history_lessparameter,55E.csv', 'static/csv/history_lessparameter,55E.csv']
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load pre-trained models
models = [tf.keras.models.load_model(model_file) for model_file in MODEL_FILES]

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        label_path = os.path.join(dir, label)
        if os.path.isdir(label_path): 
            for imagename in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, imagename))
                labels.append(label)
    return image_paths, labels

def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, color_mode='grayscale')
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

def generate_classification_reports(y_true_categorical, y_pred, target_names):
    report = classification_report(y_true_categorical, y_pred, target_names=target_names)
    return report

def plot_confusion_matrix(y_true_categorical, y_pred, target_names, filename):
    cf_matrix = confusion_matrix(y_true_categorical, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()

def generate_graph(csv_file):
    history_df = pd.read_csv(csv_file)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['accuracy'], mode='lines', name='Train'), row=1, col=1)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['val_accuracy'], mode='lines', name='Validation'), row=1, col=1)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['loss'], mode='lines', name='Train'), row=1, col=2)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['val_loss'], mode='lines', name='Validation'), row=1, col=2)
    fig.update_layout(title="Training History", xaxis_title="Epochs", yaxis_title="Value")
    graph_html = fig.to_html(full_html=False)
    return graph_html

def generate_report(model_file, train_dir, test_dir, csv_file):
    train = pd.DataFrame()
    train['image'], train['label'] = createdataframe(train_dir)
    test = pd.DataFrame()
    test['image'], test['label'] = createdataframe(test_dir)
    train_features = extract_features(train['image']) 
    test_features = extract_features(test['image'])
    x_train = train_features/255.0
    x_test = test_features/255.0
    le = LabelEncoder()
    le.fit(train['label'])
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])
    y_train_categorical = to_categorical(y_train, num_classes=7)
    y_test_categorical = to_categorical(y_test, num_classes=7)
    model = tf.keras.models.load_model(model_file)
    y_train_pred_prob = model.predict(x_train)
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)
    y_test_pred_prob = model.predict(x_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    train_report = generate_classification_reports(y_train, y_train_pred, categories)
    test_report = generate_classification_reports(y_test, y_test_pred, categories)
    plot_confusion_matrix(y_train, y_train_pred, categories, '/tmp/train_confusion_matrix.png')
    plot_confusion_matrix(y_test, y_test_pred, categories, '/tmp/test_confusion_matrix.png')
    return train_report, test_report, '/tmp/train_confusion_matrix.png', '/tmp/test_confusion_matrix.png'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        button_value = request.form.get("button-value")
        if button_value == 'live':
            return render_template('video_detection.html')
        elif button_value == 'graph':  
            return render_template('index.html') 
    return render_template('landing.html')

# Route for the live video
@app.route('/live/')
def video():
    return Response(generate_video(Capture()), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/graph1/')
def graph_report1():
    graph_html = generate_graph(CSV_FILES[0])
    train_report, test_report, train_matrix, test_matrix = generate_report(MODEL_FILES[0], IMAGE_DIRS[0], IMAGE_DIRS[1])
    return render_template('graph1.html', graph_html=graph_html, train_report=train_report, test_report=test_report,
                           train_matrix=train_matrix, test_matrix=test_matrix)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)
