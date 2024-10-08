# views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image

# Define paths to different models for each type

CT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/chest_cancer_model.keras')
MRI_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/brain_tumor_mri_model.keras')
XRAY_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/chest_xray_model.keras')

# Load the models
mri_model = load_model(MRI_MODEL_PATH)
xray_model = load_model(XRAY_MODEL_PATH)
ct_model = load_model(CT_MODEL_PATH)

# Define class labels for each type
# Define class labels corresponding to your model's output
MRI_CLASSES = {
    0: 'glioma',
    1: 'meningioma',
    2: 'notumor',
    3: 'pituitary'
}
XRAY_CLASSES = {
    0: 'Normal',
    1: 'Bacteria',
    2: 'Virus'
}

CT_CLASSES = {
    0: 'adenocarcinoma',
    1: 'large.cell.carcinoma',
    2: 'normal',
    3: 'squamous.cell.carcinoma'
}

def home(request):
    return render(request, 'index.html')

# Function for predicting MRI scans
def upload_mri(request):
    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        return predict_scan(image_file, mri_model, MRI_CLASSES)
    return JsonResponse({'error': 'Invalid request'})

# Function for predicting X-ray scans
def upload_xray(request):
    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        return predict_scan(image_file, xray_model, XRAY_CLASSES)
    return JsonResponse({'error': 'Invalid request'})

# Function for predicting CT scans
def upload_ct(request):
    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        return predict_scan(image_file, ct_model, CT_CLASSES)
    return JsonResponse({'error': 'Invalid request'})

def predict_scan(image_file, model, class_labels):
    """Helper function for making predictions on uploaded MRI scans."""
    # Load and preprocess the image
    image = Image.open(image_file)

    # Ensure the image has 3 channels (RGB). Convert if it's grayscale.
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to match the model input
    image = image.resize((150, 150))
    img_array = np.array(image)

    # Reshape the array for the model
    img_array = img_array.reshape(1, 150, 150, 3)

    # Normalize the image (optional but recommended for CNN models)
    img_array = img_array / 255.0

    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class_idx = prediction.argmax()  # Get the index of the highest probability
    predicted_class = class_labels[predicted_class_idx]

    # Return the prediction as a JSON response
    return JsonResponse({'result': predicted_class})
