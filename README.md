
# CIFAR-10 Image Classification

## Overview
This project is an end-to-end solution for classifying images from the CIFAR-10 dataset. It includes a Python-based Jupyter notebook for training and saving the model, and a Streamlit app for real-time image classification. The CIFAR-10 dataset contains 60,000 images from 10 different classes.

---

## Features
- Train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.
- Save and load the trained model for future use.
- Classify uploaded images in real time using the Streamlit app.
- Display classification results with confidence scores.

---

## Project Structure
```
project/
├── model/
│   └── cifar10_model.h5      # Trained Keras model
├── app.py                     # Streamlit app code
├── cifar10_train.ipynb        # Jupyter Notebook for training the model
├── requirements.txt           # Dependencies for the project
└── README.md                  # Documentation
```

---

## Installation and Setup

### Prerequisites
- Python 3.7+
- pip or conda installed

### Steps
1. Clone the repository or download the files.
2. Navigate to the project directory:
   ```bash
   cd project
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Training the Model

The training script is provided in the `cifar10_train.ipynb` Jupyter notebook. The steps are as follows:

1. **Dataset Loading**:
   - CIFAR-10 dataset is loaded using `tensorflow.keras.datasets`.
2. **Model Architecture**:
   - A Sequential CNN model with the following layers:
     - Two convolutional layers with ReLU activation and max-pooling.
     - Flatten layer followed by a dense layer.
     - Output layer with softmax activation.
3. **Compilation**:
   - Optimizer: Adam
   - Loss: Sparse Categorical Crossentropy
   - Metrics: Accuracy
4. **Training**:
   - The model is trained for 20 epochs with a batch size of 64.
5. **Model Saving**:
   - Save the trained model to the `model/` directory:
     ```python
     model.save('model/cifar10_model.h5')
     ```

---

## Running the Streamlit App

The Streamlit app (`app.py`) enables real-time classification of images.

### Steps
1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
2. Upload an image through the browser interface.
3. The app will classify the image and display the predicted class along with confidence scores.

---

## Code Explanation

### Training Notebook (`cifar10_train.ipynb`)
- **Imports**:
  ```python
  import tensorflow as tf
  from tensorflow.keras import layers, models
  import matplotlib.pyplot as plt
  ```
- **Model Definition**:
  ```python
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax')
  ])
  ```
- **Training**:
  ```python
  model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
  history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
  ```

### Streamlit App (`app.py`)
- **Load Model**:
  ```python
  model = tf.keras.models.load_model('model/cifar10_model.h5')
  ```
- **File Uploader**:
  ```python
  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
  ```
- **Prediction**:
  ```python
  predictions = model.predict(image_array)
  predicted_class = np.argmax(predictions, axis=1)[0]
  ```

---

## Dependencies
Add the following to `requirements.txt`:
```
streamlit
tensorflow
numpy
pillow
matplotlib
```

---

## Enhancements
1. **Batch Upload**: Allow multiple images to be uploaded for classification.
2. **Visualization**: Add Grad-CAM visualization for interpreting CNN predictions.
3. **Deployment**: Host the app on platforms like Streamlit Sharing or Heroku.
4. **Additional Models**: Extend support to other datasets or pre-trained models (e.g., ResNet, VGG).

---

## Acknowledgments
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
