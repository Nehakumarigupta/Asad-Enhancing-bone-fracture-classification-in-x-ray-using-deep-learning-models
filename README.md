# Enhancing Bone Fracture Detection in X-ray Images Using Deep Learning Model

## Overview

Enhancing Bone Fracture Detection in X-ray Images Using Deep Learning Model is a web-based application developed using Flask. The system is designed to detect and classify different types of bone fractures from X-ray images using deep learning-based feature extraction and machine learning classification.

The project uses the VGG19 deep learning model for extracting image features and a trained Random Forest classifier for predicting the fracture type. It also includes user authentication, MySQL database integration, and an AI chatbot powered by Google Gemini API.

## Key Features

- Secure user registration and login system
- X-ray image upload functionality
- Bone fracture type prediction
- VGG19-based deep feature extraction
- Random Forest-based classification
- MySQL database integration
- AI chatbot support using Google Gemini API
- Web-based interface built with Flask
- Session-based user management

## Technologies Used

- Python
- Flask
- MySQL
- TensorFlow
- Keras
- VGG19
- Scikit-learn
- Random Forest
- Joblib
- NumPy
- Pillow
- Google Generative AI
- HTML
- CSS
- JavaScript

## System Workflow

1. The user registers or logs in to the web application.
2. The user uploads an X-ray image.
3. The image is preprocessed and resized.
4. VGG19 extracts important features from the uploaded X-ray image.
5. The extracted features are scaled using a trained scaler.
6. The Random Forest model predicts the fracture type.
7. The predicted result is displayed to the user.
8. The user can also interact with the AI chatbot for assistance.

## Fracture Classes

The system can classify the following fracture types:

- Avulsion fracture
- Comminuted fracture
- Fracture Dislocation
- Greenstick fracture
- Hairline Fracture
- Impacted fracture
- Longitudinal fracture
- Oblique fracture
- Pathological fracture
- Spiral Fracture

## Database Setup

Create a MySQL database using the following SQL script:

```sql
DROP DATABASE IF EXISTS `animals`;
CREATE DATABASE `animals`;
USE `animals`;

CREATE TABLE `users` (
    `id` INT PRIMARY KEY AUTO_INCREMENT,
    `name` VARCHAR(1000),
    `email` VARCHAR(1000),
    `password` VARCHAR(225)
);


## **Installation and Setup** 🛠️

1. Clone the Repository
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
2. Install Required Dependencies
pip install flask mysql-connector-python numpy tensorflow scikit-learn joblib pillow google-generativeai werkzeug
3. Configure MySQL Database
Update the database connection in the Python file if required:

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database="animals"
)

**4. Configure Gemini API Key**

Add your Google Gemini API key in the Python file:

GEMINI_API_KEY = "your-api-key-here"
Note: Do not expose your real API key in a public GitHub repository.

**5. Add Required Model Files**
Make sure the following trained model files are available in the project directory:

vgg19_random_forest.joblib
vgg19_scaler.joblib

**6. Run the Application**
python app.py

**7. Open in Browser**
http://127.0.0.1:5000/
Project Modules
User Authentication
The application allows users to register and log in using their name, email, and password. User details are stored in a MySQL database.

## **Fracture Prediction**
The prediction module accepts an X-ray image as input, preprocesses the image, extracts features using VGG19, and predicts the fracture type using a trained Random Forest classifier.

## **AI Chatbot**
The chatbot module uses Google Gemini API to provide AI-based responses to user queries. Chat history is maintained using Flask sessions.

## **Applications**
Medical image analysis learning projects
Bone fracture classification research
Deep learning-based healthcare applications
Academic and final-year project demonstrations
## **Limitations**
The system depends on the quality of uploaded X-ray images.
Prediction accuracy depends on the training dataset and trained model.
The application is intended for educational and research purposes only.
It should not be used as a replacement for professional medical diagnosis.
## **Future Scope**
Improve model accuracy using a larger medical image dataset
Add doctor/admin dashboard
Store prediction history in the database
Deploy the application on a cloud platform
Add support for more medical image categories
Improve security using password hashing and environment variables
## **Disclaimer**
This project is developed for educational and research purposes only. The prediction results should not be considered as professional medical advice. Always consult a qualified medical professional for diagnosis and treatment.
