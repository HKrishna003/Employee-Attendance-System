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

from fastapi.responses import StreamingResponse, JSONResponse
from starlette.staticfiles import StaticFiles


app = FastAPI()

cap = cv2.VideoCapture(0)

# Parent Directory of dataset
data_upload = "D:/SREC/Data"

base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

file_path = "D:/SREC/Demo_4.xlsx"

# To add a new user to parent directory
# @app.post("/add_new_user")
# async def add_user_name(name:str, file: UploadFile = File(...)):
#     new_user_folder = os.path.join(data_upload, name)
#     try:
#         os.makedirs(new_user_folder)
#         file_path = os.path.join(new_user_folder,file.filename)
#         try:
#             with open(file_path, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error creating folder: {str(e)}")
#     return JSONResponse(content={"message": "New User Added successfully!"})

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
        if check == "out":
            person_out["name"] = person_name
        return person_name


@app.get("/in")
async def entry_in_2():
    df = pd.read_excel("D:/SREC/T3.xlsx")
    c_t = time.ctime()
    
    name = "Name"
    if name in df.columns:
        old = person_in["name"] in df['Name'].values
        if old:
            df.loc[df["Name"]==person_in["name"],"In"] = df["In"].fillna('').astype(str) + ", " + c_t
            print("Success")
    else:
        new_entry = {"Name": person_in["name"], "In": c_t, "Out": "", "Work": "", "Attendance": "", "OT": ""}
        new_df = pd.DataFrame([new_entry]) 
        df = pd.concat([df, new_df], ignore_index=True)
    
    df.to_excel("D:/SREC/T3.xlsx", index=False)
    return {"Name": person_in["name"], "Status": "Checked In!", "Time": c_t}


@app.get("/out")
async def out():
    file_path = "D:/SREC/T3.xlsx"
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
 
    
    df.loc[df[c2] == v2, u2] = df.loc[df[c2] == v2, u2].fillna('').apply(lambda x: n2 if x == '' else x + ", " + n2)
    # work(person_out["name"])
    # attendance()
    # ot()

    df.to_excel(file_path, index=False)
    
    
    return {"Name": person_out["name"], "Status": "Checked Out!", "Time": out_time}
    
    
@app.post("/live")
async def live_feed(new_user: str):
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
        
    c = 0  

    print("Press 'c' to capture an image, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

    # Display capture count on screen
        cv2.putText(frame, f"Captures: {c}/5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the live video feed
        cv2.imshow("Webcam - Press 'c' to capture, 'q' to quit", frame)

    # Capture key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and c < 5:  # If 'c' is pressed, capture image
            c += 1
            new_user_folder = os.path.join(data_upload, new_user)
            os.makedirs(new_user_folder, exist_ok=True)  # Create user folder

            # Generate filename
            filename = f"{new_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(new_user_folder, filename)

            # Save the image
            cv2.imwrite(filepath, frame)
            print(f"Image saved: {filename}")

        elif key == ord('q'):  # Quit program
            print("Exiting...")
            break

# Release resources
    cap.release()
    cv2.destroyAllWindows()
    return {"Name": new_user, "message": "Face captured successfully"}




## To calculate Work
def work(name):
    print("Calculate work")
    df = pd.read_excel("D:/SREC/T3.xlsx")
    in_time = df.loc[df["Name"] == name, "In"].values
    out_time = df.loc[df["Name"] == name, "Out"].values
    
    in_list = [x.strip() for x in str(in_time[0]).split(",") if x.strip()]
    out_list = [x.strip() for x in str(out_time[0]).split(",") if x.strip()]

    # Pair timestamps index-wise
    matched_pairs = list(zip(in_list, out_list))
    
    print(matched_pairs[0])

work("Natalie Portman")