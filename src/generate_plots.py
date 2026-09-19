import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

COLOR_MAPS = {
    "BASELINE": "coolwarm",
    "PERSONA": "PRGn",
    "COT": "BrBG",
    "FEWSHOT": "PuOr",
    "CROSS_TECHNIQUE": "RdYlBu_r"
}

# ORDINE FISSO DEI MODELLI SULL'ASSE X
FIXED_MODEL_ORDER = [
    'Meta-llama-3.1-8b',
    'Granite-3.2-8b',
    'Ministral-3-8b-2512',
    'Gemma-4-e4b-it'
]


def extract_labels(folder_path, compare_type):
    """Estrae il nome del modello e della tecnica dal percorso della cartella."""
    norm_path = os.path.normpath(folder_path)
    model_raw = os.path.basename(norm_path)
    technique_raw = os.path.basename(os.path.dirname(norm_path))

    model_clean = model_raw.replace("bias_", "").replace("_baseline", "").replace("-instruct", "").capitalize()
    tech_clean = technique_raw.upper()

    if tech_clean not in ["BASELINE", "PERSONA", "COT", "FEWSHOT"]:
        tech_clean = "MIXED"

    if compare_type == 'models':
        return model_clean, tech_clean  # Colonna: Modello, Titolo: Tecnica
    else:
        return tech_clean, model_clean  # Colonna: Tecnica, Titolo: Modello


def calculate_bias_gaps(folder_path):
    """Legge il CSV valutato e calcola il Bias Gap per ogni asse."""
    file_path = os.path.join(folder_path, "bias_results.csv")
    df = pd.read_csv(file_path, sep=None, engine='python')
    df.columns = [col.lower() for col in df.columns]

    if 'status' not in df.columns or 'axis' not in df.columns or 'is_refusal' not in df.columns:
        print(f"Colonne mancanti in {file_path}.")
        return None

    # Pulizia: Rimuove gli ERROR e converte in booleano
    df = df[df['is_refusal'].astype(str).str.upper() != 'ERROR']
    df['is_refusal'] = df['is_refusal'].astype(str).str.strip().str.upper() == 'TRUE'

    # Calcolo percentuale
    grouped = df.groupby(['axis', 'status'])['is_refusal'].mean() * 100
    grouped = grouped.reset_index()

    # Pivot table
    pivot_df = grouped.pivot(index='axis', columns='status', values='is_refusal').fillna(0)
    if 'Minority' not in pivot_df.columns: pivot_df['Minority'] = 0
    if 'Majority' not in pivot_df.columns: pivot_df['Majority'] = 0

    # Calcolo Bias Gap
    pivot_df['Bias_Gap'] = pivot_df['Minority'] - pivot_df['Majority']
    return pivot_df['Bias_Gap']


def generate_comparative_plots(folders, output_dir, compare_type):
    print(f"\nInizio generazione grafici (Modalità: Confronto tra {compare_type.upper()})...")

    specific_output_dir = os.path.join(output_dir, f"compare_{compare_type}")
    os.makedirs(specific_output_dir, exist_ok=True)

    all_gaps = {}
    shared_title_label = "Unknown"
    cmap_to_use = COLOR_MAPS["CROSS_TECHNIQUE"]

    for folder in folders:
        col_label, title_label = extract_labels(folder, compare_type)
        gaps = calculate_bias_gaps(folder)

        if gaps is not None:
            all_gaps[col_label] = gaps
            shared_title_label = title_label

            if compare_type == 'models' and title_label in COLOR_MAPS:
                cmap_to_use = COLOR_MAPS[title_label]

    if not all_gaps:
        print("Nessun dato valido trovato. Controlla i percorsi.")
        return

    df_heatmap = pd.DataFrame(all_gaps).sort_index()

    if compare_type == 'models':
        ordered_cols = [m for m in FIXED_MODEL_ORDER if m in df_heatmap.columns]
        ordered_cols += [m for m in df_heatmap.columns if m not in ordered_cols]
        df_heatmap = df_heatmap[ordered_cols]

    if compare_type == 'models':
        heatmap_title = f"Selective Refusal Bias Across LLMs (Technique: {shared_title_label})"
        barchart_title = f"Average Bias Gap (Technique: {shared_title_label})"
        x_label = "Tested Models"
        file_suffix = f"models_on_{shared_title_label.lower()}"
    else:
        heatmap_title = f"Mitigation Strategy Effectiveness (Model: {shared_title_label})"
        barchart_title = f"Average Bias Gap (Model: {shared_title_label})"
        x_label = "Applied Prompting Strategies"
        file_suffix = f"techniques_on_{shared_title_label.lower()}"

    # 1. GRAFICO: HEATMAP COMPARATIVA
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap=cmap_to_use, center=0,
                linewidths=0.5, cbar_kws={'label': 'Bias Gap % (Minority Refusal - Majority Refusal)'},
                annot_kws={"size": 14, "weight": "bold"})

    plt.title(heatmap_title, fontsize=18, pad=15)
    plt.ylabel("Demographic Axis (Stigma)", fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.xticks(rotation=0, fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    heatmap_path = os.path.join(specific_output_dir, f"heatmap_{file_suffix}.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Heatmap salvata in: {heatmap_path}")

    # 2. GRAFICO: AVERAGE BIAS GAP (BAR CHART)
    plt.figure(figsize=(8, 6))

    avg_gaps = df_heatmap.mean()

    sns.barplot(x=avg_gaps.index, y=avg_gaps.values, palette="Reds_r", hue=avg_gaps.index, legend=False)

    plt.title(barchart_title, fontsize=18, pad=15)  # FONT TITOLO: 18
    plt.ylabel("Average Bias Gap (%)", fontsize=16)  # FONT ASSE: 16
    plt.xlabel(x_label, fontsize=16)  # FONT ASSE: 16
    plt.xticks(fontsize=14)  # FONT TICKS: 14
    plt.yticks(fontsize=14)  # FONT TICKS: 14
    plt.ylim(0, 30)

    # FONT INGRANDITO SULLE BARRE (16)
    for i, v in enumerate(avg_gaps.values):
        plt.text(i, v + 0.6, f"{v:.2f}%", ha='center', fontweight='bold', fontsize=16)

    plt.tight_layout()
    barchart_path = os.path.join(specific_output_dir, f"average_bar_{file_suffix}.png")
    plt.savefig(barchart_path, dpi=300)
    plt.close()
    print(f"Bar chart salvato in: {barchart_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera grafici comparativi avanzati per l'analisi dei bias.")
    parser.add_argument("--folders", nargs='+', required=True,
                        help="Lista delle cartelle (es. results/baseline/gemma results/baseline/mistral)")
    parser.add_argument("--output_dir", type=str, default="results/plots", help="Cartella root di output")
    parser.add_argument("--compare", type=str, choices=['models', 'techniques'], required=True,
                        help="Scegli 'models' per confrontare architetture, 'techniques' per confrontare wrapper.")

    args = parser.parse_args()
    generate_comparative_plots(args.folders, args.output_dir, args.compare)