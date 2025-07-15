import os
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.contrib import messages
# from pyresparser import ResumeParser
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from .models import UserDetails



def index(request):

    
    return render(request, 'index.html')

def register(request):
    if request.method == "POST":
        name = request.POST['name']
        email = request.POST['email']
        password = request.POST['password']
        c_password = request.POST['c_password']
        if password == c_password:
            if UserDetails.objects.filter(email=email).exists():
                return render(request, 'register.html', {'message': 'User with this email already exists'})
            new_user = UserDetails(name=name, email=email, password=password)
            new_user.save()
            return render(request, 'login.html', {'message': 'Successfully Registered!'})
        return render(request, 'register.html', {'message': 'Password and Confirm Password do not match!'})
    return render(request, 'register.html')

def login(request):
    if request.method == "POST":
        email = request.POST['email']
        password1 = request.POST['password']

        try:
            user_obj = UserDetails.objects.get(email=email)
        except UserDetails.DoesNotExist:
            return render(request, 'login.html', {'message': 'Invalid Username or Password!'})

        password2 = user_obj.password
        if password1 == password2:
            return redirect('home')
        else:
            return render(request, 'login.html', {'message': 'Invalid Username or Password!'})
    return render(request, 'login.html')

# from django.shortcuts import render, redirect
# from tensorflow.keras.applications import DenseNet121
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# # Load the models (place these at the top of your views.py)
# densenet_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(128, 128, 3), pooling='avg')
# mobilenet_model = load_model('mobilenet_classifier.h5')

# # Define class indices and labels
# class_indices = {'A+': 0, 'A-': 1, 'B+': 2, 'B-': 3, 'AB+': 4, 'AB-': 5, 'O+': 6, 'O-': 7, 'unknown': 8}
# class_labels = {v: k for k, v in class_indices.items()}

# def predict_with_mobilenet(image_path):
#     img = image.load_img(image_path, target_size=(128, 128))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     features = densenet_model.predict(img_array)
#     prediction = mobilenet_model.predict(features)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     return class_labels[predicted_class]



# def home2(request):
#     predicted_label = None
#     image_url = None

#     if request.method == "POST":
#         image_file = request.FILES.get('image')

#         # Validate that an image file is provided
#         if image_file and image_file.content_type.startswith('image/'):
#             upload_dir = os.path.join(settings.BASE_DIR, 'uploaded_images')
#             if not os.path.exists(upload_dir):
#                 os.makedirs(upload_dir)

#             file_path = os.path.join(upload_dir, image_file.name)
#             with open(file_path, 'wb+') as destination:
#                 for chunk in image_file.chunks():
#                     destination.write(chunk)

#             predicted_label = predict_with_mobilenet(file_path)
#             image_url = f"/uploaded_images/{image_file.name}"

#             # Redirect to the same view with prediction results to prevent form resubmission
#             return render(request, 'home.html', {'predicted_label': predicted_label, 'image_url': image_url})
#         else:
#             # Handle case where the uploaded file is not a valid image
#             return render(request, 'home.html', {'error_message': 'Please upload a valid image file.'})

#     return render(request, 'home2.html', {'predicted_label': predicted_label, 'image_url': image_url})



from django.shortcuts import render, redirect
from PIL import Image
import numpy as np
import tensorflow as tf
from transformers import ViTFeatureExtractor, TFViTForImageClassification
import os

# Load the pre-trained ViT model and feature extractor
vit_model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=9)
vit_model.load_weights('vit_model_weights.h5')

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Define class indices and labels 
class_labels = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'unknown']

# Function to preprocess and predict with ViT model
def predict_with_vit(image_path):
    # Preprocess the image
    img = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode
    img = img.resize((224, 224))  # Resize to the input size expected by the ViT model
    img_array = np.array(img)
    pixel_values = feature_extractor(images=img_array, return_tensors="tf").pixel_values

    # Make the prediction
    prediction_logits = vit_model(pixel_values).logits
    predicted_class = np.argmax(prediction_logits, axis=-1)[0]

    # Return the predicted class label
    return class_labels[predicted_class]

# The updated home2 view that uses the ViT model 
def home2(request):
    predicted_label = None
    image_url = None

    if request.method == "POST":
        image_file = request.FILES.get('image')

        # Validate that an image file is provided
        if image_file and image_file.content_type.startswith('image/'):
            upload_dir = os.path.join(settings.BASE_DIR, 'uploaded_images')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            file_path = os.path.join(upload_dir, image_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            # Use the ViT model to predict the class label
            predicted_label = predict_with_vit(file_path)
            image_url = f"/uploaded_images/{image_file.name}"

            # Redirect to the same view with prediction results to prevent form resubmission
            return render(request, 'home.html', {'predicted_label': predicted_label, 'image_url': image_url})
        else:
            # Handle case where the uploaded file is not a valid image
            return render(request, 'home.html', {'error_message': 'Please upload a valid image file.'})

    return render(request, 'home2.html', {'predicted_label': predicted_label, 'image_url': image_url})

def home(request):
    return render(request, 'home.html')

