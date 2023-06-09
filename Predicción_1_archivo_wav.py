# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 20:05:37 2023

@author: anami
"""

import numpy as np
import librosa
import tensorflow as tf
#from tensorflow import keras

# Cargar modelo previamente entrenado
model = tf. keras.models.load_model(r"C:\Users\anami\.spyder-py3\nq_fold9.h5")

# Definir parámetros para extracción de características
def _windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size // 2)
frames=41
window_size = 512 * (frames - 1)
bands = 60

# Cargar archivo .wav
file_path = "C:/Users/anami/.spyder-py3/UrbanSound8K/music2.wav"
sound_clip, sr = librosa.load(file_path, sr=None, mono=True)

# Inicializar matriz de características
features = np.zeros((1, 41, 79, 2))

# Extraer características para cada ventana de audio
for (start, end) in _windows(sound_clip, window_size):
    if len(sound_clip[start:end]) == window_size:
        signal = sound_clip[start:end]
        stft = np.abs(librosa.stft(signal))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr).T
        melspec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=512, hop_length=512, n_mels=bands)
        logspec = librosa.power_to_db(melspec, ref=np.max).T
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr).T
        feature_mix = np.hstack((logspec, chroma, contrast))
        features[0] = feature_mix[..., np.newaxis]  # Agregar dimensión extra

# Hacer predicción con el modelo
prediction = model.predict(features)

# Imprimir etiqueta de la predicción
label_map = {0: 'Air', 1: 'carhorn', 2: 'children', 3: 'dog', 4: 'drilling', 5: 'engine',
             6: 'gun', 7: 'jack', 8: 'siren', 9:'street_music'} # mapeo de etiquetas a clases
predicted_label = np.argmax(prediction)
print('El archivo {} es de la clase {}'.format(file_path, label_map[predicted_label]))

