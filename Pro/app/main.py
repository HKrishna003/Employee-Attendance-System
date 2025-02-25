from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pickle
from sklearn.neighbors import KNeighborsClassifier
import cv2
import tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pandas as pd
import time
from datetime import datetime

app = FastAPI()

# Parent Directory of dataset
data_upload = "D:/SREC/Data"

base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# To add a new user to parent directory
@app.post("/add_new_user")
async def add_user_name(name:str, file: UploadFile = File(...)):
    new_user_folder = os.path.join(data_upload, name)
    try:
        os.makedirs(new_user_folder)
        file_path = os.path.join(new_user_folder,file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating folder: {str(e)}")
    return JSONResponse(content={"message": "New User Added successfully!"})


upload = "uploads"
isdir = os.path.isdir(upload)
if isdir:
    pass
else:
    os.makedirs(upload)

person_in = {"name":""}
person_out = {"name":""}

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize for ResNet50
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = base_model.predict(img)
    return features.flatten() 

## Entry and Exit
@app.post("/in_out")
async def test_user(check: str, file: UploadFile = File(...)):
    # if action in ["entry" or "exit"]:
        file_path = os.path.join(upload, file.filename)
        with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
        
        # # Save uploaded file
        # with open(file_path, "wb") as buffer:
        #     buffer.write(file.file.read())

        with open("D:/SREC/face_features_2.pkl", "rb") as f:
            data = pickle.load(f)

        features = data["features"]
        labels = data["labels"]
        label_encoder = data["label_encoder"]

        # Train a KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
        knn.fit(features, labels)
        
        feature_vector = extract_features(file_path)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Predict Class
        prediction = knn.predict(feature_vector)
        person_name = label_encoder.inverse_transform(prediction)[0]
        if check == "in":
            person_in["name"] = person_name
        else:
            person_out["name"] = person_name
        return person_name

@app.get("/in")
async def entry_in():
    file_path = "D:/SREC/Demo_3.xlsx"
    # try:
    #     df = pd.read_excel(file_path)  # Load existing data
    # except FileNotFoundError:
    #     df = pd.DataFrame()  # Create new DataFrame if file doesn't exist
       
    df = pd.read_excel(file_path) 
    # To get the check-in time
    c_t = time.ctime()
    
    new_entry = {"Name":person_in["name"], "In":c_t, "Out":"", "Work":""}
    new_df = pd.DataFrame([new_entry])  # Convert new data to DataFrame
    df = pd.concat([df, new_df], ignore_index=True)  # Append data
    
    df.to_excel(file_path, index=False)
    return {"Name": person_in["name"], "Status": "Checked In!", "Time": c_t}


@app.get("/out")
async def out():
    file_path = "D:/SREC/Demo_3.xlsx"
    df = pd.read_excel(file_path)
    
    c_t = time.ctime()
    
    v1 = person_out["name"]
    c1 = "Name"
    u1 = "In"  # Column to update
    out_time = c_t
   
    v2 = person_out["name"]
    c2 = "Name"
    u2 = "Out"
    u3 = "Work"
    n2 = c_t
    
    in_time = df.loc[df[c1] == v1, u1].values[0]
    
    format_str = "%a %b %d %H:%M:%S %Y"  # Example: 'Mon Feb 26 14:30:15 2024'
    dt1 = datetime.strptime(in_time, format_str)
    dt2 = datetime.strptime(out_time, format_str)

# Calculate the difference in hours
    time_difference = dt2 - dt1
    w = time_difference.total_seconds() / 3600
    w = "{:.3f}".format(w)
    
    df.loc[df[c2] == v2, u2] = n2
    df.loc[df[c2] == v2, u3] = w
    df.to_excel(file_path, index=False)
    return {"Name": person_in["name"], "Status": "Checked Out!", "Time": out_time}

    