# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:21:53 2023

@author: anami
"""

import pyaudio
import wave

# Configuración de la grabación
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "grabacion.wav"

# Inicializar PyAudio
audio = pyaudio.PyAudio()

# Obtener el índice del micrófono en el puerto jack de 3.5 mm
mic_index = None
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if "3.5mm" in info["name"]:
        mic_index = info["index"]
        break

# Verificar si se encontró el micrófono
if mic_index is None:
    print("No se encontró un micrófono en el puerto jack de 3.5 mm.")
    audio.terminate()
    exit(1)


# Abrir un flujo de audio para la grabación
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print("Grabando audio...")
frames = []

# Grabar audio durante 5 segundos
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Grabación completada.")

# Detener y cerrar el flujo de audio
stream.stop_stream()
stream.close()
audio.terminate()

# Guardar la grabación en un archivo WAV
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print("Grabación guardada en: {}".format(WAVE_OUTPUT_FILENAME))
