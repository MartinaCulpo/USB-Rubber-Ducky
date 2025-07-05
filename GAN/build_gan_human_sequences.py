import pandas as pd

# === Parametri ===
input_file = "sampled_balanced_dataset_labeled.csv"
output_file = "gan_human_sequences.csv"
events_per_sequence = 10  # 10 eventi = 20 valori (hold + flight)

# === 1. Carica dati
print(" Caricamento dati...")
df = pd.read_csv(input_file)
df_human = df[df['source'] == 'human'].copy()

# === 2. Rimuove righe con valori nulli in Hold o Flight
df_human = df_human.dropna(subset=["Hold", "Flight"]).reset_index(drop=True)

# === 3. Assegna sentence_id ogni blocco di 10 eventi
total_rows = len(df_human)
valid_rows = (total_rows // events_per_sequence) * events_per_sequence
df_human = df_human.iloc[:valid_rows]
df_human["sentence_id"] = df_human.index // events_per_sequence

print(f"  Totale righe usate: {valid_rows}")
print(f" Totali sequenze generate: {df_human['sentence_id'].nunique()}")

# === 4. Costruisce le sequenze
sequences = []

for sid, group in df_human.groupby("sentence_id"):
    values = []
    for _, row in group.iterrows():
        values.append(row["Hold"])
        values.append(row["Flight"])
    sequences.append(values)

# === 5. Crea DataFrame finale
columns = []
for i in range(1, events_per_sequence + 1):
    columns.extend([f"hold_{i}", f"flight_{i}"])

df_seq = pd.DataFrame(sequences, columns=columns)
df_seq.to_csv(output_file, index=False)

print(f"\n File '{output_file}' generato con successo.")
print(f" Righe finali: {len(df_seq)} sequenze da {events_per_sequence} eventi.")
