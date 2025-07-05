import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# === CONFIG ===
SEQUENCE_LENGTH = 20         # 10 coppie hold/flight
NOISE_DIM = 100              # Dimensione del vettore casuale z
BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 0.0002
DATA_PATH = "gan_human_sequences.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CREA FOLDER PER MODELLI ===
os.makedirs("gan_models", exist_ok=True)

# === 1. LOAD E NORMALIZZA I DATI ===
print(" Caricamento dati...")
df = pd.read_csv(DATA_PATH)
data = df.values.astype(np.float32)

# Normalizzazione Min-Max su ciascuna colonna
data_min = data.min(axis=0)
data_max = data.max(axis=0)
data_norm = (data - data_min) / (data_max - data_min + 1e-8)

tensor_data = torch.tensor(data_norm, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(tensor_data), batch_size=BATCH_SIZE, shuffle=True)

# === 2. MODELLI ===
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(NOISE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, SEQUENCE_LENGTH),
            nn.Sigmoid()  # output normalizzato [0, 1]
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(SEQUENCE_LENGTH, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE)
opt_D = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE)

# === 3. TRAINING LOOP ===
print(" Inizio training GAN...")
g_losses, d_losses, d_accuracies = [], [], []

for epoch in range(1, EPOCHS + 1):
    g_loss_total, d_loss_total, acc_total = 0, 0, 0

    for real_batch, in dataloader:
        real_batch = real_batch.to(DEVICE)
        batch_size = real_batch.size(0)

        # === Real and Fake labels ===
        real_labels = torch.ones((batch_size, 1), device=DEVICE)
        fake_labels = torch.zeros((batch_size, 1), device=DEVICE)

        # === Train Discriminator ===
        z = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
        fake_data = G(z).detach()
        d_real = D(real_batch)
        d_fake = D(fake_data)
        d_loss = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # === Accuracy del Discriminatore ===
        with torch.no_grad():
            real_preds = (d_real > 0.5).float()
            fake_preds = (d_fake < 0.5).float()
            correct = real_preds.sum() + fake_preds.sum()
            d_accuracy = correct / (2 * batch_size)

        # === Train Generator ===
        z = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
        fake_data = G(z)
        d_pred = D(fake_data)
        g_loss = loss_fn(d_pred, real_labels)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()
        acc_total += d_accuracy.item()

    g_losses.append(g_loss_total / len(dataloader))
    d_losses.append(d_loss_total / len(dataloader))
    d_accuracies.append(acc_total / len(dataloader))

    if epoch % 25 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f" Epoch {epoch}/{EPOCHS} | D loss: {d_loss_total:.4f} | G loss: {g_loss_total:.4f} | D acc: {acc_total / len(dataloader) * 100:.2f}%")

# === 4. PLOT LOSSES ===
plt.figure(figsize=(8,5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(d_accuracies, label='D Accuracy')
plt.title("GAN Training Metrics")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("gan_models/training_loss_plot.png")
plt.show()

# === 5. SALVATAGGI FINALI ===
torch.save(G.state_dict(), "gan_models/generator.pth")
np.save("gan_models/data_min.npy", data_min)
np.save("gan_models/data_max.npy", data_max)

print("\n Training completato e modello salvato!")
print(" Generatore: gan_models/generator.pth")
print(" Grafico:    gan_models/training_loss_plot.png")
print(" Scaling:    gan_models/data_min.npy / data_max.npy")
