# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:01:29 2023
@author: anami
"""

import numpy as np
import librosa
#import tensorflow as tf
#from tensorflow import keras
import tflite_runtime.interpreter as tflite
#from tflite_runtime.interpreter import Interpreter
import pyaudio
import wave
import time

# Cargar modelo previamente entrenado
#model = tf. keras.models.load_model(r"C:\Users\anami\.spyder-py3\nq_fold9.h5")
interpreter = tflite.Interpreter(model_path="nq_fold9.tflite")
interpreter.allocate_tensors()

# Obtener los índices de las entradas y salidas del modelo
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Definir parámetros para extracción de características
def _windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size // 2)
frames=41
window_size = 512 * (frames - 1)
bands = 60

# Imprimir Menú
# Imprimir el borde superior del menú
print("╔" + "═" * 56 + "╗")
# Imprimir las opciones del menú
print("║              Menú principal                            ║")
print("║                                                        ║")
print("║ 1. Para grabar solo 1 muestra de 5s                    ║")
print("║ 2. Para grabar indefinidamente                         ║")
print("║                                                        ║")
print("║                                                        ║")
print("║ 0. Salir                                               ║")
# Imprimir el borde inferior del menú
print("╚" + "═" * 56 + "╝")

# Pedir al usuario que ingrese una opción
opcion = input("Seleccione una opción: ")

# Realizar una acción en función de la opción seleccionada
if opcion == "1":
    # Configuración de la grabación de audio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 4
    OUTPUT_FILENAME = "grabacion.wav"
    #DEVICE_NAME = "edgetpu-audio-card:0" # Índice del dispositivo de hardware (0,0)

    # Inicializar PyAudio
    p = pyaudio.PyAudio()

    # Obtener información del dispositivo
    DEVICE_INFO = p.get_device_info_by_index(1)
    # Abrir el flujo de audio desde el dispositivo de hardware específico
    # Comprobar si se admite el número de canales solicitado
    if CHANNELS > DEVICE_INFO["maxInputChannels"]:
        CHANNELS = DEVICE_INFO["maxInputChannels"]
    print(f"Grabando con {CHANNELS} canal(es)")

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
                #input_device_index=DEVICE_INFO["index"])

    print("Grabando audio...")

    frames = []

    # Grabar audio en trozos
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Grabación completada.")

    # Detener el flujo de audio
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Guardar la grabación en un archivo WAV
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Grabación guardada en {OUTPUT_FILENAME}\n")

    time.sleep(3)

    print("\nContinuación del programa después de 3 segundos\n")

    # Cargar archivo .wav
    file_path = "grabacion.wav"
    sound_clip, sr = librosa.load(file_path, sr=None, mono=True)

    # Inicializar matriz de características
    features = np.zeros((1, 41, 79, 2), dtype=np.float32)

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
    #prediction = model.predict(features)
    interpreter.set_tensor(input_index, features.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)

    # Imprimir etiqueta de la predicción
    label_map = {0: 'Air', 1: 'carhorn', 2: 'children', 3: 'dog', 4: 'drilling', 5: 'engine',
                 6: 'gun', 7: 'jack', 8: 'siren', 9:'street_music'} # mapeo de etiquetas a clases
    predicted_label = np.argmax(prediction)
    print('El archivo {} es de la clase {}'.format(file_path, label_map[predicted_label]))
elif opcion == "2":
    print("Ha seleccionado la opción 2")

