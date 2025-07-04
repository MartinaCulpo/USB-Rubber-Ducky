import time
import random
from pynput.keyboard import Key, Controller
import numpy as np


keyboard = Controller()

# Delay for a given number of milliseconds
def delay(ms):
    time.sleep(ms / 1000.0)

# Convert float in [0,1) to int in [min_val, max_val]
def get_jitter(min_val, max_val, prng_fn, seed):
    value = prng_fn(seed=seed, size=1)
    if isinstance(value, list) or isinstance(value, np.ndarray):
        value = value[0]
    return int(min_val + value * (max_val - min_val))


# Type text with jitter between keystrokes
def jittered_type(text, min_jitter=20, max_jitter=100, prng_fn=None, seed=42):
    if prng_fn is None:
        prng_fn = prng_mersenne
    for char in text:
        keyboard.press(char)
        keyboard.release(char)
        jitter = get_jitter(min_jitter, max_jitter, prng_fn, seed)
        delay(jitter)



# Type text with jitter between keystrokes
def jittered_type_hold(text, min_jitter=20, max_jitter=100, min_hold=20, max_hold=100, prng_fn=None, seed=42):
    if prng_fn is None:
        prng_fn = prng_mersenne
    for char in text:
        keyboard.press(char)
        hold_time = get_jitter(min_hold, max_hold, prng_fn, seed)
        delay(hold_time)
        keyboard.release(char)
        jitter = get_jitter(min_jitter, max_jitter, prng_fn, seed)
        delay(jitter)


# Hold a key for a certain duration in milliseconds
def hold(key):
    keyboard.press(key)

# Explicitly release a key (if needed)
def release(key):
    keyboard.release(key)

# === PRNG Implementations ===

def prng_lcg(seed: int, a=1664525, c=1013904223, m=2**32, size=1):
    values = []
    x = seed
    for _ in range(size):
        x = (a * x + c) % m
        values.append(x / m)
    return values if size > 1 else values[0]


def prng_xoshiro128(seed: int, size=1):
    def rotl(x, k): return ((x << k) | (x >> (32 - k))) & 0xFFFFFFFF
    s = [seed, seed ^ 0x9E3779B9, seed ^ 0x243F6A88, seed ^ 0xB7E15162]

    def next_val():
        nonlocal s
        result = rotl(s[1] * 5, 7) * 9
        t = s[1] << 9
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 11)
        return result & 0xFFFFFFFF

    return [next_val() / 0xFFFFFFFF for _ in range(size)] if size > 1 else next_val() / 0xFFFFFFFF


def prng_mersenne(seed: int, size=1):
    rng = random.Random(seed)
    return [rng.random() for _ in range(size)] if size > 1 else rng.random()


def prng_pcg64(seed: int, size=1):
    rng = np.random.default_rng(seed)
    return rng.random(size) if size > 1 else rng.random()


def prng_philox(seed: int, size=1):
    rng = np.random.Generator(np.random.Philox(seed))
    return rng.random(size) if size > 1 else rng.random()



import sys

if __name__ == "__main__":
    prng_map = {
        "philox": prng_philox,
        "pcg64": prng_pcg64,
        "xoshiro128": prng_xoshiro128,
        "mersenne": prng_mersenne,
        "lcg": prng_lcg
    }

    prng_name = sys.argv[1] if len(sys.argv) > 1 else "mersenne"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    prng_fn = prng_map.get(prng_name.lower(), prng_mersenne)

    print(f"Typing using PRNG: {prng_name.upper()} with seed {seed}")
    delay(2000)

    jittered_type("I love you", min_jitter=20, max_jitter=100, prng_fn=prng_fn, seed=seed)

    print("\nHolding SHIFT...")
    keyboard.press(Key.enter)
    hold(Key.shift)
    jittered_type("I love you", min_jitter=20, max_jitter=100, prng_fn=prng_fn, seed=seed)
    release(Key.shift)

    keyboard.press(Key.enter)
    jittered_type_hold("I love you", min_jitter=20, max_jitter=100, min_hold=30, max_hold=90, prng_fn=prng_fn, seed=seed)

    print("\nHolding SHIFT...")
    keyboard.press(Key.enter)
    jittered_type_hold("I love you", min_jitter=20, max_jitter=100, min_hold=30, max_hold=90, prng_fn=prng_fn, seed=seed)
