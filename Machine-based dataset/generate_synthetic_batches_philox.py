import os
import pandas as pd
import numpy as np

def prng_philox(seed, size=1):
    return np.random.Generator(np.random.Philox(seed)).random(size)

def char_to_vk(c):
    if c.isalpha(): return ord(c.upper())
    if c.isdigit(): return ord(c)
    if c == ' ': return 32
    return None

def extract_sentences(df):
    grouped = df.groupby("source_file")
    sentences = []
    for _, group in grouped:
        chars = group['char'].dropna().tolist()
        text = ''.join([c for c in chars if len(c) == 1])
        if text.strip():
            sentences.append(text.strip())
    return sentences

if __name__ == "__main__":
    df = pd.read_csv("human_keystrokes.csv")
    sentences = extract_sentences(df)
    print(f"📥 Frasi totali: {len(sentences)}")

    min_hold = df['Hold'].quantile(0.01)
    max_hold = df['Hold'].quantile(0.99)
    min_flight = df['Flight'].quantile(0.01)
    max_flight = df['Flight'].quantile(0.99)

    epsilon_hold = 0.01 * (max_hold - min_hold)
    epsilon_flight = 0.01 * (max_flight - min_flight)

    hold_range = (max(10, min_hold - epsilon_hold), min(400, max_hold + epsilon_hold))
    flight_range = (max(10, min_flight - epsilon_flight), min(700, max_flight + epsilon_flight))

    BATCH_SIZE = 1000
    prng_name = "philox"
    num_batches = (len(sentences) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(num_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(sentences))
        batch_sentences = sentences[start:end]
        output_rows = []

        print(f"\n🔁 Batch {batch_num + 1}/{num_batches} → Frasi {start + 1}–{end}")

        for sid, sentence in enumerate(batch_sentences):
            sentence_id = start + sid
            prev_flight = 0

            for i, c in enumerate(sentence):
                vk = char_to_vk(c)
                if vk is None:
                    continue

                hold = int(hold_range[0] + prng_philox(sentence_id + i, 1)[0] * (hold_range[1] - hold_range[0]))
                flight = int(flight_range[0] + prng_philox(sentence_id + i + 1000, 1)[0] * (flight_range[1] - flight_range[0]))
                jitter = abs(flight - prev_flight)
                prev_flight = flight

                output_rows.append({
                    'VK': vk,
                    'Hold': hold,
                    'Flight': flight,
                    'char': c,
                    'jitter': jitter,
                    'label': 1,
                    'sentence_id': sentence_id + 1
                })

        out_file = f"synthetic_keystrokes_{prng_name}_batch_{batch_num + 1}.csv"
        pd.DataFrame(output_rows).to_csv(out_file, index=False)
        print(f"✅ Salvato: {out_file} ({len(output_rows)} righe)")
