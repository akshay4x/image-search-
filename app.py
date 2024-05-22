import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload and images directories exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists('static/images'):
    os.makedirs('static/images')

# Load pre-trained ResNet50 model + higher level layers
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    """Extract features from an image using ResNet50."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features

def find_similar_image(upload_path):
    try:
        uploaded_features = extract_features(upload_path)
        closest_image = None
        highest_similarity = -1

        print("Starting search for similar images in the 'images' directory.")
        for image_name in os.listdir('static/images'):
            image_path = os.path.join('static/images', image_name)
            image_features = extract_features(image_path)
            similarity = cosine_similarity(uploaded_features, image_features)[0][0]

            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_image = image_name
                print(f"Found closer image: {image_name} with similarity {similarity}")

        print(f"Most similar image found: {closest_image}")
        return closest_image

    except Exception as e:
        print(f"Error finding similar image: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part in the request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        if file:
            try:
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)
                print(f"File {filename} uploaded successfully to {upload_path}")
                return redirect(url_for('result', filename=filename))
            except Exception as e:
                flash(f"Error saving file: {e}")
                return redirect(request.url)
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    similar_image = find_similar_image(upload_path)
    if similar_image:
        return render_template('result.html', uploaded_image=filename, similar_image=similar_image)
    else:
        flash("No similar image found.")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
