import os
import time
import csv
from pynput import keyboard

output_path = r"C:\Users\utente\OneDrive - Università degli Studi di Padova\Desktop\ATCNS\Project\RUBBER DUCKY\Code\logger_output.csv"

pressed_keys = {}
events = []
last_release_time = None
start_logging = False
enter_keycode = 28  # Enter key VK (approx)
entered_keys = []

def on_press(key):
    global start_logging, last_release_time, entered_keys
    current_time = int(time.time() * 1000)
    try:
        vk = key.vk if hasattr(key, 'vk') else key.value.vk
    except AttributeError:
        return

    if vk in pressed_keys:
        return

    if not start_logging:
        if vk == enter_keycode:
            entered_keys.append(vk)
            return
        else:
            start_logging = True
            entered_keys = []

    flight_time = current_time - last_release_time if last_release_time else 0
    pressed_keys[vk] = current_time
    events.append({'VK': vk, 'press_time': current_time, 'flight_time': flight_time})

def on_release(key):
    global last_release_time
    current_time = int(time.time() * 1000)
    try:
        vk = key.vk if hasattr(key, 'vk') else key.value.vk
    except AttributeError:
        return

    if vk in pressed_keys:
        press_time = pressed_keys.pop(vk)
        hold_time = current_time - press_time
        for event in reversed(events):
            if event['VK'] == vk and 'hold_time' not in event:
                event['hold_time'] = hold_time
                break
        last_release_time = current_time

def monitor_inactivity(listener, timeout=5):
    global last_release_time
    while True:
        time.sleep(1)
        now = int(time.time() * 1000)
        if last_release_time and (now - last_release_time > timeout * 1000):
            listener.stop()
            break

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
monitor_inactivity(listener)

# Save CSV
with open(output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['VK', 'Hold Time (ms)', 'Flight Time (ms)'])
    for e in events:
        if 'hold_time' in e and e['VK'] != '':
            writer.writerow([e['VK'], e['hold_time'], e['flight_time']])
