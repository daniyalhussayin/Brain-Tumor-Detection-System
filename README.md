# Brain Tumor Detection System

A Deep Learning-based web application that detects different types of brain tumors from MRI scan images using a Convolutional Neural Network (CNN).  
Built with **TensorFlow/Keras** and deployed using **Streamlit** for a clean and interactive user experience.

---

## Features

- Upload MRI scan images
- AI-powered brain tumor classification
- Confidence score prediction
- Clean Streamlit interface
- Fast and lightweight application

---

## Tumor Classes

The model can detect:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- Pillow (PIL)
- Scikit-learn

---

## Project Structure

```
Brain-Tumor-Detection/
│
├── app.py
├── train_model.py
├── brain_tumor_model.h5
├── requirements.txt
└──  README.md

```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/daniyalhussayin/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```
