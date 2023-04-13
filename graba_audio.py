# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:21:53 2023

@author: anami
"""

import pyaudio
import wave

# Configuración de la grabación de audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
OUTPUT_FILENAME = "grabacion.wav"
DEVICE_INDEX = 0  # Índice del dispositivo de hardware (0,0)

# Inicializar PyAudio
p = pyaudio.PyAudio()

# Abrir el flujo de audio desde el dispositivo de hardware específico
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=DEVICE_INDEX)

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

print(f"Grabación guardada en {OUTPUT_FILENAME}")

