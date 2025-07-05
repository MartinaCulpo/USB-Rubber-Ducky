import os

def find_human_csv_files_by_filename(root_folder):
    human_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            # Verifica se il nome del file contiene "HUMAN" (ignorando maiuscole/minuscole) e finisce con .csv
            if "human" in file.lower() and file.lower().endswith(".csv"):
                full_path = os.path.join(root, file)
                human_files.append(full_path)
    return human_files

# === ESECUZIONE ===
if __name__ == "__main__":
    dataset_root = r"C:\Users\utente\OneDrive - Università degli Studi di Padova\Desktop\ATCNS\Project\RUBBER DUCKY\Dataset Mendeley"

    result = find_human_csv_files_by_filename(dataset_root)
    print(f" Trovati {len(result)} file CSV che hanno 'HUMAN' nel nome del file:\n")

    for path in result:
        print(path)

    # (Opzionale) salva in file .txt
    with open("lista_file_human.txt", "w") as f:
        for path in result:
            f.write(path + "\n")
