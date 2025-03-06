import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import pickle

base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Function to Extract Features
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize for ResNet50
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = base_model.predict(img)
    return features.flatten()  # Convert to 1D

# Directory Containing Face Images
dataset_path = "D:/SREC/Dataset_2"  # Folder with subfolders for each person
feature_list = []
labels = []

# Process each image

for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            feature_vector = extract_features(img_path)
            feature_list.append(feature_vector)
            labels.append(person)  # Folder name is the class label

# Convert to NumPy Array
features = np.array(feature_list)
labels = np.array(labels)



# Encode Labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)


# Save Features and Labels
with open("D:/SREC/Model/dataset_2.pkl", "wb") as f:
    pickle.dump({"features": features, "labels": encoded_labels, "label_encoder": le}, f)

print("Feature extraction updated and saved!")


