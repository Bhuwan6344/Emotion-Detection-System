import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, after_this_request
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

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Define directories
TRAIN_DIR = '/Users/sam/Downloads/Dataset/train'
TEST_DIR = '/Users/sam/Downloads/Dataset/test'

# Other global variables
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load pre-trained model
model = tf.keras.models.load_model('/Users/sam/Desktop/7th sem/web-app/backend/models/model_lessparameter,55E.h5')

# Function to create dataframe
def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        label_path = os.path.join(dir, label)
        if os.path.isdir(label_path): 
            for imagename in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, imagename))
                labels.append(label)
            print(label, "completed")
    return image_paths, labels

# Function to extract features
def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, color_mode='grayscale')
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

# Function to generate classification reports
def generate_classification_reports(y_true_categorical, y_pred, target_names):
    report = classification_report(y_true_categorical, y_pred, target_names=target_names)
    return report

# Function to plot confusion matrix
def plot_confusion_matrix(y_true_categorical, y_pred, target_names, filename):
    cf_matrix = confusion_matrix(y_true_categorical, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to avoid potential threading issues

# Route for generating classification reports and confusion matrices
@app.route('/generate_reports')
def generate_reports():
    # Create dataframes
    train = pd.DataFrame()
    train['image'], train['label'] = createdataframe(TRAIN_DIR)
    test = pd.DataFrame()
    test['image'], test['label'] = createdataframe(TEST_DIR)
    # Extract features for train and test data
    train_features = extract_features(train['image']) 
    test_features = extract_features(test['image'])
    # Normalize
    x_train = train_features/255.0
    x_test = test_features/255.0
    # Encode labels
    le = LabelEncoder()
    le.fit(train['label'])
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])
    y_train_categorical = to_categorical(y_train, num_classes=7)
    y_test_categorical = to_categorical(y_test, num_classes=7)
    # Make predictions for train and test data
    y_train_pred_prob = model.predict(x_train)
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)
    y_test_pred_prob = model.predict(x_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    # Generate classification report for train and test data
    train_report = generate_classification_reports(y_train, y_train_pred, categories)
    test_report = generate_classification_reports(y_test, y_test_pred, categories)
    # Plot confusion matrix for train and test data
    plot_confusion_matrix(y_train, y_train_pred, categories, '/tmp/train_confusion_matrix.png')
    plot_confusion_matrix(y_test, y_test_pred, categories, '/tmp/test_confusion_matrix.png')
    # Return the reports and matrices filenames
    return render_template('reports.html', train_report=train_report, test_report=test_report,
                           train_matrix='/tmp/train_confusion_matrix.png',
                           test_matrix='/tmp/test_confusion_matrix.png')

# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)
