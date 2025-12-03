
# 🐄 Animal Disease Identification using AIML (Lumpy Skin Disease Focus)

## 📌 Project Overview

This project is a **deep learning-powered web application** developed to enable **early and accurate detection of Lumpy Skin Disease (LSD) in cattle**. The system bridges the diagnostic gap for farmers and veterinarians in remote areas by providing **instant, accessible, and reliable predictions**.

The solution combines **Convolutional Neural Networks (CNN)** for disease detection with a **cloud-hosted web interface**, ensuring usability across devices.

🔗 **Live Deployment**: [Click Here](http://ec2-3-110-124-28.ap-south-1.compute.amazonaws.com:5000/) *(Replace with your latest deployment URL if changed)*

---

## 📡 System Architecture Diagram

![architecture diagram](https://github.com/user-attachments/assets/a616f2f3-05bf-420f-bb37-0c6ec53c8157)


---


## 👨‍💻 Team & Duration

* **Team Size**: 4
* **Team Members**:

  * **Sudhanshu Pendor**
  * **Dhiraj Rathod**
  * **Shivanjali Jagtap**
  * **Shravani Sakore**
* **Mentor**: Dr. Manish Giri
* **Duration**: January 2024 


---

## 🛠️ Tech Stack & Skills Used

* **Machine Learning / Deep Learning**: Convolutional Neural Networks (CNN), TensorFlow, Keras
* **Programming Languages**: Python
* **Web Development**: Flask, HTML, CSS, JavaScript
* **Cloud & Deployment**: Amazon Web Services (AWS EC2), WinSCP, PuTTY, Git
* **Other Expertise**: Image Processing, Data Augmentation, Hyperparameter Tuning, Model Deployment

---

## 🚀 Key Features

* Upload cattle images for **real-time disease detection**
* Classifies cattle as **Healthy** or **Infected with LSD**
* Provides **prediction confidence score** for better reliability
* Web application accessible from **any device via AWS cloud**
* Lightweight and responsive **frontend UI** for ease of use

---

## 👨‍🔬 My Contributions

* **Model Development**:

  * Built and fine-tuned CNN with preprocessing (resizing, normalization).
  * Applied **data augmentation** to increase dataset robustness.
  * Optimized using **hyperparameter tuning** for high accuracy.

* **Full-Stack Development**:

  * Developed **frontend UI** with HTML, CSS, JS.
  * Built **Flask-based backend API** for model inference.

* **Cloud Deployment & DevOps**:

  * Configured and launched **AWS EC2 instance (Ubuntu)**.
  * Transferred files using **WinSCP** and managed server with **PuTTY**.
  * Set up Python virtual environment and installed dependencies.
  * Configured security groups and exposed **port 5000** for web access.

---

## 📊 Outcomes

* Achieved **98%+ accuracy** on clear images.
* Delivered a **fully functional cloud-based web app** accessible in real time.
* Enabled farmers and veterinarians to receive **instant diagnostic results** with confidence scores.
* Provided disease awareness and intervention support to reduce economic losses.

---

## 📂 Project Structure

```
├── dataset/                 # Image dataset (Healthy / Infected)
├── models/                  # Saved trained models
├── static/                  # CSS, JS, and static assets
├── templates/               # HTML templates for Flask
├── app.py                   # Main Flask application
├── model_training.ipynb     # Model training Jupyter Notebook
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/lsd-detection.git
   cd lsd-detection
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows

   pip install -r requirements.txt
   ```

3. Run the Flask app:

   ```bash
   python app.py
   ```

4. Open in browser:

   ```
   http://127.0.0.1:5000/
   ```

---
<img width="735" height="373" alt="image" src="https://github.com/user-attachments/assets/5e9fab0b-e119-430a-8de1-b91a8b836d11" />


---

## 📜 License

This project is for **academic and research purposes**.
For commercial use, please contact the project mentor or contributors.


