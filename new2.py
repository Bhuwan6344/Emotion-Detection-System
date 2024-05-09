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

from backend.video import Capture, generate_video

from backend.video import *

app = Flask(__name__, static_url_path='/static')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Define directories for images and models
IMAGE_DIRS = ['/Users/sam/Downloads/Dataset/train',
              '/Users/sam/Downloads/Dataset/test']

MODEL_FILES = ['/Users/sam/Desktop/7th sem/web-app/backend/models/model_B16,E30.h5',
               '/Users/sam/Desktop/7th sem/web-app/backend/models/model_B32,E50.h5',
               '/Users/sam/Desktop/7th sem/web-app/backend/models/model_B42,E55.h5']

CSV_FILES = ['static/csv/history_B16,E30.csv', 
             'static/csv/history_B32,E50.csv',
             'static/csv/history_B42,E55.csv']

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

def generate_graph(csv_file):
    history_df = pd.read_csv(csv_file)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['accuracy'], mode='lines', name='Train'), row=1, col=1)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['val_accuracy'], mode='lines', name='Validation'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['loss'], mode='lines', name='Train'), row=1, col=2)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['val_loss'], mode='lines', name='Validation'), row=1,
                  col=2)
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
    x_train = train_features / 255.0
    x_test = test_features / 255.0
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
    return train_report, test_report

def generate_confusion_matrix(model_file, dataset_dir, le, categories):
    data = pd.DataFrame()
    data['image'], data['label'] = createdataframe(dataset_dir)
    features = extract_features(data['image'])
    x_data = features / 255.0
    y_data = le.transform(data['label'])
    y_data_categorical = to_categorical(y_data, num_classes=len(categories))

    model = tf.keras.models.load_model(model_file)
    y_pred_prob = model.predict(x_data)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_data_categorical, axis=1)

    cf_matrix = confusion_matrix(y_true, y_pred_classes)

    return cf_matrix

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
def epoch1():
    # Generate graph HTML
    graph_html = generate_graph(CSV_FILES[0])

    # Generate classification reports
    train_report, test_report = generate_report(MODEL_FILES[0], IMAGE_DIRS[0], IMAGE_DIRS[1], CSV_FILES[0])

    # Load label encoder for train data
    le_train = LabelEncoder()
    le_train.fit(createdataframe(IMAGE_DIRS[0])[1])

    # Generate confusion matrix for train data
    cf_matrix_train = generate_confusion_matrix(MODEL_FILES[0], IMAGE_DIRS[0], le_train, categories)

    # Load label encoder for test data
    le_test = LabelEncoder()
    le_test.fit(createdataframe(IMAGE_DIRS[1])[1])

    # Generate confusion matrix for test data
    cf_matrix_test = generate_confusion_matrix(MODEL_FILES[0], IMAGE_DIRS[1], le_test, categories)

    # Pass relevant data to the template
    return render_template('graph1.html', graph_html=graph_html, train_report=train_report, test_report=test_report,
                           cf_matrix_train=cf_matrix_train, cf_matrix_test=cf_matrix_test, categories=categories)
    
    
@app.route('/graph2/')
def epoch2():
    # Generate graph HTML
    graph_html = generate_graph(CSV_FILES[1])

    # Generate classification reports
    train_report, test_report = generate_report(MODEL_FILES[1], IMAGE_DIRS[0], IMAGE_DIRS[1], CSV_FILES[1])

    # Load label encoder for train data
    le_train = LabelEncoder()
    le_train.fit(createdataframe(IMAGE_DIRS[0])[1])

    # Generate confusion matrix for train data
    cf_matrix_train = generate_confusion_matrix(MODEL_FILES[1], IMAGE_DIRS[0], le_train, categories)

    # Load label encoder for test data
    le_test = LabelEncoder()
    le_test.fit(createdataframe(IMAGE_DIRS[1])[1])

    # Generate confusion matrix for test data
    cf_matrix_test = generate_confusion_matrix(MODEL_FILES[1], IMAGE_DIRS[1], le_test, categories)

    # Pass relevant data to the template
    return render_template('graph2.html', graph_html=graph_html, train_report=train_report, test_report=test_report,
                           cf_matrix_train=cf_matrix_train, cf_matrix_test=cf_matrix_test, categories=categories)

@app.route('/graph3/')
def epoch3():
    # Generate graph HTML
    graph_html = generate_graph(CSV_FILES[2])

    # Generate classification reports
    train_report, test_report = generate_report(MODEL_FILES[2], IMAGE_DIRS[0], IMAGE_DIRS[1], CSV_FILES[2])

    # Load label encoder for train data
    le_train = LabelEncoder()
    le_train.fit(createdataframe(IMAGE_DIRS[0])[1])

    # Generate confusion matrix for train data
    cf_matrix_train = generate_confusion_matrix(MODEL_FILES[2], IMAGE_DIRS[0], le_train, categories)

    # Load label encoder for test data
    le_test = LabelEncoder()
    le_test.fit(createdataframe(IMAGE_DIRS[1])[1])

    # Generate confusion matrix for test data
    cf_matrix_test = generate_confusion_matrix(MODEL_FILES[2], IMAGE_DIRS[1], le_test, categories)

    # Pass relevant data to the template
    return render_template('graph3.html', graph_html=graph_html, train_report=train_report, test_report=test_report,
                           cf_matrix_train=cf_matrix_train, cf_matrix_test=cf_matrix_test, categories=categories)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)
