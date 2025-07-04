import matplotlib.pyplot as plt
import numpy as np
import random

# === PRNG Implementations ===

def prng_philox(seed=42, size=100000):
    rng = np.random.Generator(np.random.Philox(seed))
    return rng.random(size)

def prng_pcg64(seed=42, size=100000):
    rng = np.random.default_rng(seed)
    return rng.random(size)

def prng_mt19937(seed=42, size=100000):
    rng = random.Random(seed)
    return [rng.random() for _ in range(size)]

def prng_xoshiro128(seed=42, size=100000):
    def splitmix64(x):
        z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def rotl(x, k):
        return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF

    s = [splitmix64(seed), splitmix64(seed + 1)]

    def next_xoroshiro128():
        s0, s1 = s
        result = (s0 + s1) & 0xFFFFFFFFFFFFFFFF
        s1 ^= s0
        s[0] = rotl(s0, 55) ^ s1 ^ ((s1 << 14) & 0xFFFFFFFFFFFFFFFF)
        s[1] = rotl(s1, 36)
        return result / 0xFFFFFFFFFFFFFFFF

    return [next_xoroshiro128() for _ in range(size)]

# === Generate Samples ===

seed = 42
size = 100000

generators = {
    "Philox Distribution": prng_philox(seed, size),
    "Xoroshiro128 Distribution": prng_xoshiro128(seed, size),
    "PCG64 Distribution": prng_pcg64(seed, size),
    "MT19937 Distribution": prng_mt19937(seed, size)
}

# === Plotting ===

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for ax, (title, data) in zip(axs, generators.items()):
    ax.hist(data, bins=50, density=True, color='cornflowerblue', edgecolor='black')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig("distribution_iid.png", dpi=300)
plt.show()
