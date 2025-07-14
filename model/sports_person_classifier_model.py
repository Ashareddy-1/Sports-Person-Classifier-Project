import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import pywt
import pandas as pd
import seaborn as sn
import joblib
import json
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Load Haarcascade classifiers
face_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_eye.xml')

# Function to get cropped image if 2 eyes are detected
def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print(f"No faces detected in image: {image_path}")
        return None
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            print(f"Face and eyes detected in image: {image_path}")
            return roi_color
    print(f"Less than 2 eyes detected in image: {image_path}")
    return None

# Wavelet Transform function
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

# Paths
path_to_data = "dataset"
path_to_cr_data = "dataset/cropped/"

# Create cropped folder
if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

cropped_image_dirs = []
celebrity_file_names_dict = {}

# Process each image directory
img_dirs = [entry.path for entry in os.scandir(path_to_data) if entry.is_dir()]
print("Image directories:", img_dirs)

for img_dir in img_dirs:
    count = 1
    celebrity_name = os.path.basename(img_dir)
    celebrity_file_names_dict[celebrity_name] = []

    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = os.path.join(path_to_cr_data, celebrity_name)
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder:", cropped_folder)
            cropped_file_name = celebrity_name + str(count) + ".png"
            cropped_file_path = os.path.join(cropped_folder, cropped_file_name)
            success = cv2.imwrite(cropped_file_path, roi_color)
            if success:
                print(f"Image saved: {cropped_file_path}")
                celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                count += 1
            else:
                print(f"Failed to save image: {cropped_file_path}")
        else:
            print(f"No valid cropped image found for {entry.path}")

# Regenerate `celebrity_file_names_dict` after manual cleaning
celebrity_file_names_dict = {}
for img_dir in cropped_image_dirs:
    celebrity_name = os.path.basename(img_dir)
    file_list = [entry.path for entry in os.scandir(img_dir) if entry.is_file()]
    celebrity_file_names_dict[celebrity_name] = file_list
print("Updated celebrity file names dictionary:", celebrity_file_names_dict)

# Create class dictionary
class_dict = {celebrity_name: idx for idx, celebrity_name in enumerate(celebrity_file_names_dict.keys())}
print("Class dictionary:", class_dict)

# Prepare training data
X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            print(f"Error: Unable to load training image: {training_image}")
            continue
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name])

X = np.array(X).reshape(len(X), 4096).astype(float)
print("Training data shape:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Model parameters
model_params = {
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'model__C': [1, 10, 100, 1000],
            'model__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'model__C': [1, 5, 10]
        }
    }
}

# Grid search
scores = []
best_estimators = {}
for algo, mp in model_params.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', mp['model'])])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)
print(best_estimators)

# Evaluate models
for algo in best_estimators.keys():
    print(f"{algo} test score: {best_estimators[algo].score(X_test, y_test)}")

# Confusion matrix for the best model
best_clf = best_estimators['svm']
cm = confusion_matrix(y_test, best_clf.predict(X_test))
print(cm)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Save the trained model
joblib.dump(best_clf, 'saved_model.pkl')

# Save class dictionary
with open("class_dictionary.json", "w") as f:
    f.write(json.dumps(class_dict))