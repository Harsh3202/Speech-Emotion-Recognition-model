
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model("/kaggle/working/speech_classification.h5")
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad', 'surprise']

def extract_actual_emotion(file_path):
    """Extract the actual emotion from filename (assumes format like YAF_happy_001.wav)"""
    filename = os.path.basename(file_path)
    label = filename.split('_')[-1]
    label = label.split('.')[0]
    labels.append(label.lower())
    return label

def predict_emotion(file):
    audio, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    input_data = np.expand_dims(mfcc_mean, axis=0)
    result = model.predict(input_data)[0]
    emotion = labels[np.argmax(result)]
    return emotion

if __name__ == "__main__":
    path = "/kaggle/input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/YAF_sad/YAF_book_sad.wav"
    actual = extract_actual_emotion(path)
    predicted = predict_emotion(path)

    print(f"Actual Emotion: {actual}")
    print(f"Predicted Emotion: {predicted}")
