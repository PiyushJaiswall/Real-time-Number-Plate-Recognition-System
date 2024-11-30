# **License Plate Recognition System**

This repository contains the implementation of a **License Plate Recognition System** using deep learning and computer vision techniques. The project processes vehicle images to detect license plates and extract their alphanumeric content with high accuracy.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Directory Structure](#directory-structure)
4. [Dependencies](#dependencies)
5. [How to Run](#how-to-run)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Results](#results)
8. [Future Enhancements](#future-enhancements)
9. [Acknowledgments](#acknowledgments)

---

## **Overview**

This project uses **OpenCV**, **TensorFlow**, and other Python libraries to implement a pipeline for recognizing license plates in images. The pipeline includes:

1. Detecting the license plate in an image.
2. Segmenting the characters on the plate.
3. Classifying the characters using a deep learning model.

The system was trained on a custom dataset and evaluated for accuracy, precision, recall, and F1 score.

---

## **Key Features**
- **License Plate Detection**: Uses contour detection and image preprocessing to locate plates in vehicle images.
- **Character Segmentation**: Efficiently segments individual characters from the detected license plate.
- **Deep Learning Model**: Trained a CNN using TensorFlow and Keras to classify characters.
- **End-to-End Pipeline**: From image input to license plate number extraction.
- **Interactive Visualizations**: Visualizes detected license plates, segmented characters, and model predictions.

---

## **Directory Structure**

```plaintext
📂 License_Plate_Recognition/
├── 📂 data/                   # Dataset for training and validation
│   ├── 📂 train/              # Training images organized by class
│   └── 📂 val/                # Validation images organized by class
├── 📂 logs/                   # TensorBoard logs for model training
├── 📄 ocr_model.h5            # Trained OCR model
├── 📄 LCR.ipynb  # Main implementation in Jupyter Notebook
├── 📄 README.md               # Documentation file (this file)
```
## **Dependencies**

The project requires the following libraries:

- Python 3.8+
- OpenCV
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## How to Run

### Clone the Repository:
```bash
git clone https://github.com/PiyushJaiswall/Real-time-Number-Plate-Recognition-System.git
cd License_Plate_Recognition
```
### Run the Jupyter Notebook:
- Open `LCR.ipynb` in a Jupyter Notebook environment.
- Execute the cells step by step to reproduce the pipeline.

### Visualize Results:
The notebook includes code to visualize:
- Detected license plates
- Segmented characters
- Predictions

## Model Training and Evaluation

- **Dataset**: The model was trained on a dataset of alphanumeric characters, with real-time data augmentation applied using `ImageDataGenerator`.
- **Metrics**: Accuracy, precision, recall, and F1 score were used to evaluate model performance.
- **Saved Model**: The final trained model is saved as `ocr_model.h5`.

---

## Results

### Key Achievements:
- **License Plate Detection**: Achieved robust detection across various lighting and angle conditions.
- **Character Recognition Accuracy**: Achieved an accuracy of over 99% on the validation dataset.

### Example Output:
- Detected license plate with predicted alphanumeric text.

---

## Future Enhancements
Potential improvements to the system include:
1. **Support for More Languages**: Extend the character recognition model to handle multiple languages.
2. **Real-Time Detection**: Implement real-time video processing for continuous license plate recognition.
3. **Improved Dataset**: Train the model on a larger and more diverse dataset.

## Acknowledgments

Special thanks to:
- OpenCV and TensorFlow communities for providing robust libraries.
- Datasets and tutorials that inspired this implementation.

