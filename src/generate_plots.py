import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


def clean_model_name(folder_path):
    """Estrae un nome pulito del modello dal percorso della cartella."""
    base_name = os.path.basename(os.path.normpath(folder_path))
    # Rimuove prefissi e suffissi inutili per rendere i grafici più puliti
    return base_name.replace("bias_", "").replace("_baseline", "").replace("-instruct", "").capitalize()


def calculate_bias_gaps(folder_path):
    """Legge il CSV del modello e calcola il Bias Gap per ogni asse."""
    file_path = os.path.join(folder_path, "bias_results.csv")
    if not os.path.exists(file_path):
        print(f"Salto {folder_path}: file 'bias_results.csv' non trovato.")
        return None

    # Supporto per separatori multipli (, o ;)
    df = pd.read_csv(file_path, sep=None, engine='python')

    # Normalizzazione nomi colonne (per sicurezza)
    df.columns = [col.lower() for col in df.columns]
    status_col = 'status'
    axis_col = 'axis'
    refusal_col = 'is_refusal'

    if status_col not in df.columns or axis_col not in df.columns or refusal_col not in df.columns:
        print(f"Colonne mancanti in {file_path}.")
        return None

    # Calcolo percentuale di rifiuto per asse e status
    df[refusal_col] = df[refusal_col].astype(bool)
    grouped = df.groupby([axis_col, status_col])[refusal_col].mean() * 100
    grouped = grouped.reset_index()

    # Pivot per avere Majority e Minority sulla stessa riga
    pivot_df = grouped.pivot(index=axis_col, columns=status_col, values=refusal_col).fillna(0)

    # Assicurati che le colonne esistano, altrimenti settale a 0
    if 'Minority' not in pivot_df.columns: pivot_df['Minority'] = 0
    if 'Majority' not in pivot_df.columns: pivot_df['Majority'] = 0

    # Calcolo Bias Gap
    pivot_df['Bias_Gap'] = pivot_df['Minority'] - pivot_df['Majority']

    return pivot_df['Bias_Gap']


def generate_comparative_plots(folders, output_dir):
    print("\nInizio generazione grafici comparativi...")
    os.makedirs(output_dir, exist_ok=True)

    all_gaps = {}

    for folder in folders:
        model_name = clean_model_name(folder)
        gaps = calculate_bias_gaps(folder)
        if gaps is not None:
            all_gaps[model_name] = gaps

    if not all_gaps:
        print("Nessun dato valido trovato. Controlla i percorsi.")
        return

    # Crea il DataFrame combinato (Righe: Assi, Colonne: Modelli)
    df_heatmap = pd.DataFrame(all_gaps)

    # Ordina gli assi in ordine alfabetico per coerenza
    df_heatmap = df_heatmap.sort_index()

    # ==========================================
    # 1. GRAFICO: HEATMAP COMPARATIVA
    # ==========================================
    plt.figure(figsize=(12, 8))
    # Usiamo 'coolwarm' centrato sullo 0: Blu = no bias, Rosso = alto bias a favore delle minoranze
    sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap="coolwarm", center=0,
                linewidths=0.5, cbar_kws={'label': 'Bias Gap % (Minority Refusal - Majority Refusal)'})

    plt.title("Selective Refusal Bias Gap Across LLM Architectures", fontsize=16, pad=15)
    plt.ylabel("Demographic Axis (Stigma)", fontsize=12)
    plt.xlabel("Tested Models", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, "comparative_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Heatmap salvata in: {heatmap_path}")

    # ==========================================
    # 2. GRAFICO: AVERAGE BIAS GAP (BAR CHART)
    # ==========================================
    plt.figure(figsize=(8, 6))
    avg_gaps = df_heatmap.mean().sort_values(ascending=False)

    sns.barplot(x=avg_gaps.index, y=avg_gaps.values, palette="Reds_r", hue=avg_gaps.index, legend=False)

    plt.title("Average Selective Refusal Bias Gap per LLM", fontsize=14, pad=15)
    plt.ylabel("Average Bias Gap (%)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylim(0, max(avg_gaps.values) + 5)

    # Aggiungi i valori sopra le barre
    for i, v in enumerate(avg_gaps.values):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    barchart_path = os.path.join(output_dir, "average_bias_bar.png")
    plt.savefig(barchart_path, dpi=300)
    plt.close()
    print(f"Bar chart salvato in: {barchart_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera grafici comparativi sul Bias Gap da più modelli.")
    parser.add_argument("--folders", nargs='+', required=True,
                        help="Lista delle cartelle dei modelli (es. results/bias_mistral results/bias_llama...)")
    parser.add_argument("--output_dir", type=str, default="results/plots",
                        help="Cartella dove salvare le immagini generate")

    args = parser.parse_args()
    generate_comparative_plots(args.folders, args.output_dir)