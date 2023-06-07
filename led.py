from periphery import GPIO
import time

rojo = GPIO("/dev/gpiochip0", 6, "out")  # pin 13
azul = GPIO("/dev/gpiochip2", 9, "out")  # pin 16
verde = GPIO("/dev/gpiochip4", 10, "out")  # pin 18
blanco = GPIO("/dev/gpiochip4", 12, "out")  # pin 22
amarillo = GPIO("/dev/gpiochip0", 7, "out")  # pin 29

rojo.write(True)
azul.write(True)
verde.write(True)
blanco.write(True)
amarillo.write(True)
time.sleep(5)
rojo.write(False)
azul.write(False)
verde.write(False)
blanco.write(False)
amarillo.write(False)
rojo.close()
verde.close()
azul.close()
blanco.close()
amarillo.close()
	
