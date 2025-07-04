import csv
import time
from pynput import keyboard

output_path = "logger_output.csv"

pressed_keys = {}
last_release_time = None
last_event_time = time.time()

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Virtual Key", "Character", "Hold Time (ms)", "Flight Time (ms)"])

    def get_char(key):
        try:
            return key.char
        except AttributeError:
            return str(key)  # es: Key.enter, Key.space...

    def on_press(key):
        global last_release_time, last_event_time
        current_time = time.time()
        last_event_time = current_time

        try:
            vk = key.vk if hasattr(key, 'vk') else key.value.vk
        except AttributeError:
            return

        char = get_char(key)
        flight_time = 0
        if last_release_time is not None:
            flight_time = round((current_time - last_release_time) * 1000)

        pressed_keys[vk] = (current_time, flight_time, char)

    def on_release(key):
        global last_release_time, last_event_time
        current_time = time.time()
        last_event_time = current_time

        try:
            vk = key.vk if hasattr(key, 'vk') else key.value.vk
        except AttributeError:
            return

        if vk in pressed_keys:
            press_time, flight_time, char = pressed_keys[vk]
            hold_time = round((current_time - press_time) * 1000)
            writer.writerow([vk, char, hold_time, flight_time])
            f.flush()
            del pressed_keys[vk]

        last_release_time = current_time

    print(" Recording keystrokes... (stop typing for 5s to auto-exit)")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while True:
        time.sleep(0.1)
        if time.time() - last_event_time > 5:
            print(" Inactivity timeout reached. Logger stopped.")
            listener.stop()
            break
