import time

def contador_tiempo():
    # Establecer el tiempo límite en segundos
    tiempo_limite = 30  # 30 seg

    # Obtener el tiempo de inicio
    tiempo_inicio = time.time()

    while True:
        # Verificar si se ha excedido el tiempo límite
        if time.time() - tiempo_inicio > tiempo_limite:
            print("Tiempo límite alcanzado. ¡Saliendo del programa!")
            break

        # Leer la entrada del usuario
        entrada = input("Ingresa un carácter: ")

        # Verificar si no se ha ingresado ningún carácter
        if not entrada:
            print("No se ingresó ningún carácter. ¡Saliendo del programa!")
            break

        # Si se ingresó un carácter, continuar con la ejecución del programa
        # ...
        # Aquí puedes poner el resto de tu lógica

# Llamar a la función para iniciar el contador de tiempo
contador_tiempo()
