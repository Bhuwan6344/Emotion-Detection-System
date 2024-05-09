import sys
sys.path.append(r"C:\Users\BHUWAN\Downloads\web-app (1)")

from flask import Flask, render_template, Response, request
import pandas as pd

from sklearn.metrics import classification_report

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Importing modules from the backend package
# main.py

from backend import video

# Your other code here


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Function to generate graph HTML from a CSV file
def generate_graph(csv_file):
    # Read the CSV file
    history_df = pd.read_csv(csv_file)

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))

    # Add traces for accuracy
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['accuracy'], mode='lines', name='Train'), row=1, col=1)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['val_accuracy'], mode='lines', name='Validation'), row=1, col=1)

    # Add traces for loss
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['loss'], mode='lines', name='Train'), row=1, col=2)
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['val_loss'], mode='lines', name='Validation'), row=1, col=2)

    # Update layout
    fig.update_layout(title="Training History",
                      xaxis_title="Epochs",
                      yaxis_title="Value")

    # Convert the figure to HTML content
    graph_html = fig.to_html(full_html=False)

    return graph_html

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

# Route for generating graphs
@app.route('/graph/<int:graph_num>/')
def generate_graph_route(graph_num):
    csv_file_path = f'static/csv/history_lessparameter_{graph_num}.csv'
    graph_html = generate_graph(csv_file_path)
    return render_template(f'graph{graph_num}.html', graph_html=graph_html)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=False)
