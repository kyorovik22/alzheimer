import os
from flask import Flask, render_template, request, redirect, url_for, flash
from keras.models import load_model # type: ignore
import joblib
from PIL import Image
import numpy as np
from flask import send_from_directory

# Setup Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey'

# Load the models
cnn_model = load_model('models/gres18032.h5')
svm_model = joblib.load('models/gres18032.pkl')

# Define class labels
class_labels = {0: 'AD - Alzheimer\'s Disease', 1: 'CI - Cognitive Impairment', 2: 'CN - Cognitive Normal'}

# Function to preprocess the image
def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to extract features using CNN
def extract_features(cnn_model, img_path):
    img = preprocess_image(img_path)
    features = cnn_model.predict(img)
    return features.flatten()

def classify_with_svm(features):
    # Check if predict_proba is available
    if hasattr(svm_model, 'predict_proba'):
        # Predict using SVM model
        prediction = svm_model.predict([features])
        predicted_class = int(prediction[0])
        accuracy = svm_model.predict_proba([features])[0][predicted_class]
    else:
        prediction = svm_model.predict([features])
        predicted_class = int(prediction[0])
        accuracy = None  # Accuracy is not available without predict_proba

    # Return the corresponding class label and accuracy
    return class_labels.get(predicted_class, "Unknown class"), accuracy

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image upload and prediction
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Ensure the file is an image
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Perform prediction
            features = extract_features(cnn_model, filepath)
            predicted_class, accuracy = classify_with_svm(features)

            if predicted_class == 'AD - Alzheimer\'s Disease':
                message = '''Alzheimer adalah bentuk demensia progresif yang mempengaruhi memori, pemikiran, dan perilaku. Tahap ini ditandai oleh:
                            - Penurunan signifikan dalam fungsi kognitif yang mengganggu kehidupan sehari-hari
                            - Kehilangan memori yang parah, terutama untuk informasi baru
                            - Kesulitan dalam perencanaan, pemecahan masalah, dan penyelesaian tugas-tugas familiar
                            - Perubahan kepribadian dan perilaku
                            - Pada tahap lanjut, kesulitan berbicara, menelan, dan melakukan aktivitas dasar sehari-hari
                            '''
            elif predicted_class == 'CI - Cognitive Impairment':
                message = '''MCI adalah tahap antara penurunan kognitif normal akibat penuaan dan demensia yang lebih serius. Karakteristiknya meliputi:
                            - Penurunan ringan dalam fungsi kognitif, terutama memori, yang lebih besar dari yang diharapkan untuk usia seseorang
                            - Kesulitan dengan tugas-tugas kompleks atau pemecahan masalah
                            - Masih dapat melakukan sebagian besar aktivitas sehari-hari secara mandiri
                            - Tidak semua orang dengan MCI akan berkembang menjadi demensia atau Alzheimer
                            '''
            elif predicted_class == 'CN - Cognitive Normal':
                message = '''Ini adalah kondisi fungsi kognitif yang dianggap normal atau tipikal untuk usia seseorang. Pada tahap ini:
                            - Individu memiliki kemampuan berpikir, mengingat, dan bernalar yang sesuai dengan usia mereka.
                            - Mereka dapat melakukan aktivitas sehari-hari tanpa kesulitan yang signifikan.
                            - Memori jangka pendek dan jangka panjang berfungsi dengan baik.
                            - Kemampuan pemecahan masalah dan pengambilan keputusan tetap terjaga.'''

            
            flash(f'Predicted class: {predicted_class}')
            flash(message)
            return render_template('upload.html', filepath=file.filename, message=message)
        else:
            flash('Invalid file type. Please upload an image.')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
