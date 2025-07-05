import pandas as pd
import os

# === Mappa VK → Caratteri leggibili ===
VK_TO_CHAR = {
    8: '[BACKSPACE]', 9: '[TAB]', 13: '[ENTER]', 32: ' ',
    48: '0', 49: '1', 50: '2', 51: '3', 52: '4',
    53: '5', 54: '6', 55: '7', 56: '8', 57: '9',
    **{code: chr(code) for code in range(65, 91)}
}

def parse_csv(path):
    try:
        df = pd.read_csv(path)
        df = df.rename(columns={"VK": "VK", "HT": "Hold", "FT": "Flight"})
        df["char"] = df["VK"].map(VK_TO_CHAR).fillna("[UNK]")
        df["jitter"] = df["Flight"].diff().abs().fillna(0)
        df["source_file"] = os.path.basename(path)
        return df
    except Exception as e:
        print(f"⚠️ Errore nel file {path}: {e}")
        return pd.DataFrame()

def process_batch(paths, output_file, first_batch=False):
    frames = []
    for path in paths:
        df = parse_csv(path)
        if not df.empty:
            frames.append(df)
    if frames:
        result = pd.concat(frames, ignore_index=True)
        result.to_csv(output_file, mode='w' if first_batch else 'a', header=first_batch, index=False)
        print(f"✅ Salvato batch con {len(result)} righe")

# === Esecuzione principale ===
if __name__ == "__main__":
    BATCH_SIZE = 1000
    OUTPUT_FILE = "human_keystrokes.csv"

    with open("lista_file_human.txt", "r") as f:
        paths = [line.strip() for line in f if line.strip().endswith(".csv")]

    total = len(paths)
    print(f"🔍 Trovati {total} file da processare in batch da {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch_paths = paths[i:i + BATCH_SIZE]
        print(f"⏳ Batch {i // BATCH_SIZE + 1}: {len(batch_paths)} file...")
        process_batch(batch_paths, OUTPUT_FILE, first_batch=(i == 0))

    print(f"\n🎉 Completato. Dati uniti salvati in '{OUTPUT_FILE}'")
