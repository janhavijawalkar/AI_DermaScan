🧴 DermalScan: AI_Facial Skin Aging Detection App

DermalScan is a deep learning–based web application that detects and classifies facial skin aging conditions such as Wrinkles, Dark Spots, Puffy Eyes, and Clear Skin using a pretrained EfficientNetB0 model.
The system integrates face detection, image preprocessing, classification, confidence visualization, and a user-friendly web interface.

📌 Project Overview

The goal of DermalScan is to automate facial skin aging analysis using Artificial Intelligence.
Users can upload a facial image, and the system will:

Detect the face

Identify the skin condition

Display confidence scores

Show annotated output images

Allow result downloads


🧠 Problem Statement

The objective is to develop a deep learning-based system that can detect and classify facial aging signs—such as wrinkles, dark spots, puffy eyes, and clear skin—using a pretrained EfficientNetB0 model. The pipeline includes face detection using Haar Cascades, custom preprocessing and data augmentation, and classification with percentage predictions. A web-based frontend will enable users to upload images and visualize aging signs with annotated bounding boxes and labels. 

📂 Dataset Description

Source: Kaggle

Classes:

Wrinkles

Dark Spots

Puffy Eyes

Clear Skin

📊 Dataset Preparation

Initial dataset had ~300 images per class

Clear Skin images were manually cleaned and verified

Data Augmentation applied

Final balanced dataset:

409 images per class

<img width="1536" height="1024" alt="ChatGPT Image Dec 28, 2025, 03_24_41 PM" src="https://github.com/user-attachments/assets/0a3f0ae3-517b-4abc-821d-7a8940b2a016" />


🔗 Dataset Link:
[📥 Download Dataset from Google Drive]

👉(https://drive.google.com/drive/folders/1oDO54S6tc-GX5w61X9AvWG-z8dgvZoKy?usp=sharing))

🧩 Project Modules (End-to-End Workflow)
🔹 Module 1: Data Collection & Cleaning

Collected facial skin images from Kaggle

Removed noisy and mislabeled images

Manual verification for Clear Skin class

🔹 Module 2: Data Preprocessing

Image resizing to 224 × 224

RGB conversion

Normalization using EfficientNet preprocessing

Data augmentation:

Rotation

Horizontal flipping

Zoom

🔹 Module 3: Face Detection

Used Haar Cascade Classifier

Detected face regions before classification

Fallback to full-image analysis if no face detected

🔹 Module 4: Feature Extraction

Used EfficientNetB0 (pretrained on ImageNet)

Automatic extraction of:

Texture features

Spatial facial patterns

Global Average Pooling used to reduce dimensionality

🔹 Module 5: Model Architecture

EfficientNetB0 backbone

Transfer Learning applied

Fully Connected layers for classification

Output layer with Softmax activation

🔹 Module 6: Model Training

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Dataset split:

Training: 80%

Validation: 20%

Techniques used:

Early Stopping

Model Checkpointing

🔹 Module 7: Model Evaluation

Achieved ~94% validation accuracy

Stable training with minimal overfitting

Reliable confidence predictions
<img width="767" height="613" alt="Screenshot 2025-12-28 190229" src="https://github.com/user-attachments/assets/e99302bd-b1d0-4cbb-842a-6e8774c57d39" />


🔹 Module 8: Web Application (UI)

Built using Streamlit

Features:

Image upload

Annotated output image

Confidence display

Probability distribution

Processing time display

Download results (CSV & image)
<img width="1884" height="781" alt="Screenshot 2025-12-28 193025" src="https://github.com/user-attachments/assets/5d97a864-361c-48d8-a5ff-b96b2f00455b" />
<img width="1620" height="779" alt="Screenshot 2025-12-28 193105" src="https://github.com/user-attachments/assets/929b57b6-964c-47de-ac1f-cfaed33771b8" />
<img width="1795" height="835" alt="Screenshot 2026-01-01 195741" src="https://github.com/user-attachments/assets/8950b9a8-45e6-406b-9cff-2bc53104251e" />




🖥️ Technologies Used

Programming Language: Python

Deep Learning: TensorFlow, Keras

Model: EfficientNetB0

Computer Vision: OpenCV

Web Framework: Streamlit

Data Handling: NumPy, Pandas

📈 Results

Validation Accuracy: ~94%

Clear and confident predictions

Real-time inference capability

User-friendly interface

⚠️ Challenges

Noisy dataset

Class imbalance

Lighting variations in images

Limitations of Haar Cascade for complex angles

🚧 Limitations

Limited dataset size

Performance depends on image quality

Haar Cascade struggles with side faces and poor lighting

🔮 Future Scope

Add more skin conditions (Acne, Pigmentation, Fine Lines)

Multi-label skin condition detection

Replace Haar Cascade with MTCNN / YOLO

Personalized skincare recommendations

Mobile application integration

🔗 Project Links

GitHub Repository:
👉 https://github.com/janhavijawalkar/AI_DermaScan

Dataset Link:
👉 https://drive.google.com/drive/folders/1oDO54S6tc-GX5w61X9AvWG-z8dgvZoKy?usp=sharing


Janhavi Jawalkar
B.Tech – Computer Science Engineering
