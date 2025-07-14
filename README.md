🏋️‍♂️ Sports Person Classifier Web App

A full-stack web application that classifies images of sports persons using a trained machine learning model. Built using HTML/CSS/JS for the frontend and Python (Flask or similar) for the backend.

📁 Project Structure

├── UI/                          # Frontend code

│   ├── images/                  # Static image resources

│   ├── test_images/            # Sample test images

│   ├── app.html                # Main HTML interface

│   ├── app.css                 # Custom styles

│   ├── app.js                  # App logic and upload handlers

│   ├── dropzone.min.css/js     # DropzoneJS for drag & drop uploads

│
├── model/                      # Model-related resources

│   ├── dataset/                # Dataset folder (likely used for training)

│   ├── test_images/            # Sample images to test model predictions

│   ├── opencv/haarcascades/    # Haar cascades for face detection

│   ├── class_dictionary.json   # Label-to-classname mapping

│   ├── saved_model.pkl         # Trained ML model

│   ├── sports_person_classifier_model.py  # Training or testing script

│   ├── requirements.txt        # Python dependencies
│
├── server/                     # Backend server logic

│   ├── artifacts/              # Model & dictionary for production

│   │   ├── class_dictionary.json

│   │   ├── saved_model.pkl

│   ├── opencv/haarcascades/    # Haar cascades for live detection

│   ├── test_images/            # Backend test images

│   ├── b64.txt                 # Base64 image input or encoded images

│   ├── server.py               # Main API/backend server script

│   ├── util.py                 # Utility functions

│   ├── wavelet.py              # Custom wavelet feature extraction

🚀 Features

📸 Upload images via a drag-and-drop web UI.

🔍 Detect and classify sports persons using a pretrained model.

🧠 Uses Haar cascades and wavelet transforms for feature extraction.

🧪 Includes test images and model artifacts for local testing.

🧪 Setup Instructions

1. Clone the Repository

2. Create a Virtual Environment
3. 
python -m venv venv

source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Run the Server
   
cd server

python server.py

The server will typically run at http://127.0.0.1:5000.

🌐 Running the Frontend (UI)

Open UI/app.html in your browser.

Drag and drop an image into the upload box.

View the predicted class of the sports person.

📂 Important Files

saved_model.pkl – Trained ML model (SVM or similar).

class_dictionary.json – Class label mappings.

wavelet.py – Feature extraction using wavelets.

server.py – Main backend app serving predictions.

🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


