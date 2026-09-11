import pandas as pd
import argparse
import os


def extract_stratified_sample(input_path, output_path, samples_per_group, seed):
    print(f"Caricamento dataset da: {input_path}")

    if not os.path.exists(input_path):
        print(f"Errore: Il file {input_path} non esiste.")
        return

    df = pd.read_csv(input_path)

    # Raggruppiamo per TEMPLATE (prompt_id) e STATUS
    # 16 template * 2 status = 32 gruppi. Selezioniamo 3 per gruppo = 96 righe esatte.
    try:
        df_shuffled = df.sample(frac=1, random_state=seed)

        # 2. Raggruppiamo e prendiamo semplicemente le prime 3 righe (che ora sono casuali)
        df_sample = df_shuffled.groupby(['prompt_id', 'status'], as_index=False).head(samples_per_group)

    except ValueError as e:
        print(f"Errore di campionamento: {e}")
        return

    # Mescoliamo le righe finali
    df_sample = df_sample.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Salvataggio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sample.to_csv(output_path, index=False)

    print(f"\nCampione stratificato estratto con successo in: {output_path}")
    print("-" * 50)
    print("STATISTICHE DEL CAMPIONE:")
    print(f"Totale righe: {len(df_sample)}")

    # Stampa in modo sicuro solo le colonne che esistono effettivamente
    if 'status' in df_sample.columns:
        print(f"\nDistribuzione Status:\n{df_sample['status'].value_counts().to_string()}")
    elif 'Status' in df_sample.columns:
        print(f"\nDistribuzione Status:\n{df_sample['Status'].value_counts().to_string()}")

    if 'prompt_id' in df_sample.columns:
        print("-" * 50)
        print("Distribuzione per Template (prompt_id):")
        print(df_sample['prompt_id'].value_counts().to_string())

    if 'axis' in df_sample.columns:
        print("-" * 50)
        print("Distribuzione per Asse:")
        print(df_sample['axis'].value_counts().to_string())
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrae un campione stratificato dal dataset.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--samples_per_group", type=int, default=3,
                        help="Righe per combinazione Template/Status (default 3 -> 96 righe tot)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    extract_stratified_sample(args.input, args.output, args.samples_per_group, args.seed)