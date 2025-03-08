import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras import models, layers


# Define a function to read in an image file and convert it to a numpy array
def read_image(file_path, size=(256, 256)):
    with Image.open(file_path) as img:
        img = img.resize(size)
        img = img.convert('RGB')
        img_data = np.asarray(img)
    return img_data

parent_dir = r"D:\SREC\Dataset_3"
print(os.listdir(parent_dir))
# Get all subdirectories (each representing a class)
class_dirs = sorted([d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))])

# Initialize lists for images and labels
all_images = []
all_labels = []

# Loop through each class directory and assign labels dynamically
for idx, class_name in enumerate(class_dirs):
    class_path = os.path.join(parent_dir, class_name)
    images = [read_image(os.path.join(class_path, file)) for file in os.listdir(class_path)]
    labels = [idx] * len(images)  # Assign label based on index

    all_images.extend(images)
    all_labels.extend(labels)
cl = len(class_dirs)
images = all_images
labels = all_labels

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

import numpy as np
def normalize_image(img):
    # Convert the image to float data type
    img = img.astype('float32')

    # Normalize the image pixels to have zero mean and unit variance
    img -= np.mean(img)
    img /= np.std(img)

    return img

X_train = np.array([normalize_image(img) for img in X_train])
X_test = np.array([normalize_image(img) for img in X_test])
X_val = np.array([normalize_image(img) for img in X_val])



cnn = models.Sequential([
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(cl, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, np.array(Y_train),  epochs=15,verbose=2)

print(os.listdir(parent_dir))
cnn.save(r"D:\SREC\CNN_model\Model.h5")