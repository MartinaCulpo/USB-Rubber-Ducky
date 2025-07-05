import pandas as pd

# === Parametri ===
input_file = "sampled_balanced_dataset_labeled.csv"
output_file = "sampled_balanced_dataset_small.csv"
samples_per_class = 5000  # numero di righe per ciascuna sorgente

# === Carica dataset completo ===
print(f"📥 Caricamento '{input_file}'...")
df = pd.read_csv(input_file)

# === Lista delle 5 sorgenti attese ===
sources = ['human', 'mersenne', 'pcg64', 'philox', 'xoroshiro128']
df_list = []

# === Estrai righe bilanciate da ciascuna sorgente ===
for source in sources:
    subset = df[df['source'] == source]
    if len(subset) < samples_per_class:
        print(f"⚠️ Warning: '{source}' ha solo {len(subset)} righe disponibili.")
        df_sampled = subset
    else:
        df_sampled = subset.sample(n=samples_per_class, random_state=42)
    
    df_list.append(df_sampled)
    print(f"✅ Aggiunto: {source} ({len(df_sampled)} righe)")

# === Unisci tutto e salva ===
df_small = pd.concat(df_list).sample(frac=1, random_state=42).reset_index(drop=True)
df_small.to_csv(output_file, index=False)

print(f"\n✅ Dataset ridotto salvato in '{output_file}'")
print("📊 Distribuzione finale:")
print(df_small['source'].value_counts())
