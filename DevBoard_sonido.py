# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:01:29 2023
@author: anami
"""

import numpy as np
import librosa
import tflite_runtime.interpreter as tflite
import pyaudio
import wave
from periphery import GPIO
import requests
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

#Encendido de las luces LED
def encender_luces(predicted_label):
    rojo = GPIO("/dev/gpiochip0", 6, "out")  # pin 13
    azul = GPIO("/dev/gpiochip2", 9, "out")  # pin 16
    verde = GPIO("/dev/gpiochip4", 10, "out")  # pin 18
    blanco = GPIO("/dev/gpiochip4", 12, "out")  # pin 22
    amarillo = GPIO("/dev/gpiochip0", 7, "out")  # pin 29
    if predicted_label == 0:
        #Aire Acondicionado: Rojo, Verde, Amarillo
        rojo.write(True)
        verde.write(True)
        amarillo.write(True)
        time.sleep(3)
        rojo.write(False)
        verde.write(False)
        amarillo.write(False)
        rojo.close()
        verde.close()
        amarillo.close()
    elif predicted_label == 1:
        #Claxon: Azul, Blanco
        azul.write(True)
        blanco.write(True)
        time.sleep(3)
        azul.write(False)
        blanco.write(False)
        azul.close()
        blanco.close()
    elif predicted_label == 2:
        #Niños: Rojo, Verde, Blanco
        rojo.write(True)
        verde.write(True)
        blanco.write(True)
        time.sleep(3)
        rojo.write(False)
        verde.write(False)
        blanco.write(False)
        rojo.close()
        verde.close()
        blanco.close()        
    elif predicted_label == 3:
        #Ladrido de perro: Rojo, Amarillo, Blanco
        rojo.write(True)
        amarillo.write(True)
        blanco.write(True)
        time.sleep(3)
        rojo.write(False)
        amarillo.write(False)
        blanco.write(False)
        rojo.close()
        amarillo.close()
        blanco.close() 
    elif predicted_label == 4:
        #Taladradora: Azul, Verde, Blanco
        azul.write(True)
        blanco.write(True)
        verde.write(True)
        time.sleep(3)
        azul.write(False)
        blanco.write(False)
        verde.write(False)
        azul.close()
        blanco.close()
        verde.close()
    elif predicted_label == 5:
        #Motor: Azul, Verde
        azul.write(True)
        verde.write(True)
        time.sleep(3)
        azul.write(False)
        verde.write(False)
        azul.close()
        verde.close()
    elif predicted_label == 6:
        #Disparo: Rojo, Verde, Azul
        rojo.write(True)
        verde.write(True)
        azul.write(True)
        time.sleep(3)
        rojo.write(False)
        verde.write(False)
        azul.write(False)
        rojo.close()
        verde.close()
        azul.close()
    elif predicted_label == 7:
        #Martillo: Azul, Amarillo, Blanco
        azul.write(True)
        amarillo.write(True)
        blanco.write(True)
        time.sleep(3)
        azul.write(False)
        amarillo.write(False)
        blanco.write(False)
        azul.close()
        amarillo.close()
        blanco.close() 
    elif predicted_label == 8:
        #Sirena: Rojo, Amarillo
        rojo.write(True)
        amarillo.write(True)
        time.sleep(3)
        rojo.write(False)
        amarillo.write(False)
        rojo.close()
        amarillo.close()
    elif predicted_label == 9:
        #Música Callejera: Blanco, Amarillo
        blanco.write(True)
        amarillo.write(True)
        time.sleep(3)
        blanco.write(False)
        amarillo.write(False)
        blanco.close()
        amarillo.close()
        
# Definir obtención de la IP
def get_ip():
    response = requests.get('https://api64.ipify.org?format=json').json()
    return response["ip"]

#Definir parámetros de ubicación
def get_location():
    ip_address = get_ip()
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = {
        "IP": ip_address,
        "Ciudad": response.get("city"),
        "Región": response.get("region"),
        "País": response.get("country_name"),
        "Latitud": response.get("latitude"),
        "Longitud": response.get("longitude")
    }
    return location_data

while True:
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

        print(f"Grabación guardada en {OUTPUT_FILENAME}\n")

        time.sleep(1)

        print("\nContinuación del programa después de 1 segundos\n")
        
        start_time=time.time()
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
        label_map = {0: 'Aire_acondicionado', 1: 'Claxon', 2: 'Voces_de_niños', 3: 'Ladrido', 4: 'Taladradora', 5: 'Motor', 
                     6: 'Disparo', 7: 'Martillo neumático', 8: 'Sirena', 9:'Música_callejera'} # mapeo de etiquetas a clases
        predicted_label = np.argmax(prediction)
        print('El archivo {} es de la clase {}'.format(file_path, label_map[predicted_label]))
        encender_luces(predicted_label)
        end_time = time.time()
        print("\n\n")
        elapsed_time = end_time - start_time
        
        # Imprimir el tiempo empleado en realizar la predicción del sonido detectado
        print(f"Tiempo de ejecución (predicción del modelo): {elapsed_time} segundos\n\n")
        print("\n\n")
        
        # Imprimir la ubicación del sonido detectado
        #print(get_location())
        #print("\n\n")
        
    elif opcion == "2":
        count=1
        print("\n\n")
        print("\nHa seleccionado la opción 2\n")
        print("\n\n")
        # Pedir al usuario que ingrese las veces a repetir el bucle
        veces = int(input("¿Cuántas veces desea repetir el bucle?: "))
        while count<=veces:
            count=count+1
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

            print(f"Grabación guardada en {OUTPUT_FILENAME}\n")

            time.sleep(1)

            print("\nContinuación del programa después de 1 segundos\n")

            start_time=time.time()
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
            label_map = {0: 'Aire_acondicionado', 1: 'Claxon', 2: 'Voces_de_niños', 3: 'Ladrido', 4: 'Taladradora', 5: 'Motor',
                         6: 'Disparo', 7: 'Martillo neumático', 8: 'Sirena', 9:'Música_callejera'} # mapeo de etiquetas a clases
            predicted_label = np.argmax(prediction)
            print('El archivo {} es de la clase {}'.format(file_path, label_map[predicted_label]))
            encender_luces(predicted_label)
            end_time = time.time()
            print("\n\n")
            elapsed_time = end_time - start_time
            
            # Imprimir el tiempo empleado en realizar la predicción del sonido detectado
            print(f"Tiempo de ejecución (predicción del modelo): {elapsed_time} segundos\n\n")
            print("\n\n")
            
            # Imprimir la ubicación del sonido detectado
            #print(get_location())
            #print("\n\n")
            
       # if count==veces:
        count=1
        respuesta = input("Presione 9 para salir, o cualquier otra tecla para continuar: ")
        if respuesta == "9":
            break
    elif opcion =="0":
        print("\nFin del programa\n")
        break
    else:
        print("El número ingresado no es válido, intente de nuevo")

