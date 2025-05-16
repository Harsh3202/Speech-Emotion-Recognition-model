
# 🗣️ Speech Emotion Classification 🎧

This project is a speech emotion recognition system trained on the [Toronto Emotional Speech Set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess). It predicts emotional states such as *happy, sad, angry*, etc., from audio files using MFCC features and a deep learning model built with TensorFlow/Keras.

---

## 📁 Project Structure

```
│──emotion_model.h5              # Trained Keras model
├── example.wav                 # Example audio file for testing
├── predict.py                  # Script to predict emotion from audio
├──Speech_Emotion_Classification.ipynb  # Jupyter Notebook
├── README.md                   # Project description
```

---

## 🚀 Features

- ✅ Predicts 8 emotional classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `ps`, `sad`, `surprise`
- 🎙️ Uses MFCC (Mel-Frequency Cepstral Coefficients) for feature extraction
- 🧠 Deep learning model with multiple Dense layers
- 📊 High classification accuracy on validation set
- 📁 Dataset: [TESS Dataset from Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

---

## 🛠️ Installation & Requirements

Install the necessary packages:

```bash
pip install librosa tensorflow numpy
```

---

## 🧪 How to Use

1. Ensure your trained model is saved in the as `emotion_model.h5`.
2. Add or record a test audio file (e.g., `example.wav`).
3. Run the prediction script:

```bash
python predict.py
```

You will see output like:

```
Actual Emotion: happy
Predicted Emotion: happy
```

---

## 🧠 Model Architecture

The model was trained using:

- **Input**: MFCC features from .wav files
- **Layers**:
  - 3 × Dense (hidden layers)
  - Dropout for regularization
  - Final Dense for classification
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

---

## 📈 Results

- Achieved high accuracy on validation data.
- Effective on clean, spoken audio.

---

## 📚 Files Included

- `Speech_Emotion_Classification.ipynb`: Full training and preprocessing pipeline
- `predict.py`: Minimal script to test model on new audio files
- `emotion_model.h5`: Trained model
- `example.wav`: Example speech file for demo

---

## 📸 Screenshots

![alt text](images/your_image.png)`)*

---
