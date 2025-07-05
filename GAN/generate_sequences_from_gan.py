import torch
import numpy as np
import pandas as pd
import os

# === CONFIG ===
NOISE_DIM = 100
SEQUENCE_LENGTH = 20  # 10 coppie hold+flight
NUM_SAMPLES = 5000    # quante righe generate
GENERATOR_PATH = "gan_models/generator.pth"
DATA_MIN_PATH = "gan_models/data_min.npy"
DATA_MAX_PATH = "gan_models/data_max.npy"
OUTPUT_CSV = "gan_generated_sequences.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. LOAD GENERATOR MODEL ===
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(NOISE_DIM, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, SEQUENCE_LENGTH),
            torch.nn.Sigmoid()  # output in range [0,1]
        )

    def forward(self, z):
        return self.model(z)

print(" Caricamento generatore...")
G = Generator().to(DEVICE)
G.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
G.eval()

# === 2. LOAD NORMALIZATION PARAMETERS ===
data_min = np.load(DATA_MIN_PATH)
data_max = np.load(DATA_MAX_PATH)

# === 3. GENERA SEQUENZE ===
print(f" Generazione di {NUM_SAMPLES} sequenze GAN...")
with torch.no_grad():
    noise = torch.randn(NUM_SAMPLES, NOISE_DIM).to(DEVICE)
    generated = G(noise).cpu().numpy()

# === 4. DE-NORMALIZZA ===
generated_denorm = generated * (data_max - data_min + 1e-8) + data_min
generated_denorm = np.clip(generated_denorm, 0, None)  # niente valori negativi

# === 5. SALVA CSV ===
columns = []
for i in range(1, 11):
    columns.append(f"hold_{i}")
    columns.append(f"flight_{i}")

df = pd.DataFrame(generated_denorm, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f" File '{OUTPUT_CSV}' generato con successo!")
