from periphery import GPIO
import time

led = GPIO("/dev/gpiochip0", 6, "out")  # pin 13

led.write(True)
time.sleep(5)
led.write(False)
led.close()
	