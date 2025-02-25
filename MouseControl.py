from pynput.mouse import Button, Controller
import time

# Create a mouse controller object
mouse = Controller()

# Move the cursor to position (200, 200)
mouse.position = (200, 200)

# Click at the current cursor position
#mouse.click(Button.left, 1)

time.sleep(1)
# Move the cursor relative to its current position

for i in range(0, 1600, 100):
    mouse.move(100, 0)
    time.sleep(1)
