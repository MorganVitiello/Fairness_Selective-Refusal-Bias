import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def run_metrics():
    parser = argparse.ArgumentParser(description="Calculate Bias Gap metrics and generate charts.")
    parser.add_argument("--input_csv", type=str, required=True, help="Percorso al file evaluated_generations.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Errore: File non trovato {args.input_csv}")
        return

    # ==========================================
    # GESTIONE NOMI E PERCORSI (AUTOMATICA)
    # ==========================================
    result_dir = os.path.dirname(args.input_csv)
    model_name = os.path.basename(result_dir)  # Es. gemma-4-e4b-it
    technique_name = os.path.basename(os.path.dirname(result_dir))  # Es. persona, cot, baseline

    # Creiamo un'etichetta pulita per i titoli
    tested_info = f"{model_name.upper()} (Technique: {technique_name.upper()})"

    summary_file = os.path.join(result_dir, f"summary_report.txt")
    chart_file = os.path.join(result_dir, f"chart_refusal_rates.png")

    df = pd.read_csv(args.input_csv)

    # ==========================================
    # PULIZIA DATI (DATA CLEANING)
    # ==========================================
    # Rimuoviamo eventuali righe andate in errore durante la valutazione
    df = df[df['is_refusal'].astype(str).str.upper() != 'ERROR']

    # Convertiamo rigorosamente in Booleani True/False (1/0)
    df['is_refusal'] = df['is_refusal'].astype(str).str.strip().str.upper() == 'TRUE'

    stats = []
    for axis in df['axis'].unique():
        sub = df[df['axis'] == axis]

        # Gestione di possibili errori/valori mancanti
        maj_sub = sub[sub['status'] == 'Majority']['is_refusal'].dropna()
        min_sub = sub[sub['status'] == 'Minority']['is_refusal'].dropna()

        maj_rate = maj_sub.mean() * 100 if not maj_sub.empty else 0
        min_rate = min_sub.mean() * 100 if not min_sub.empty else 0

        stats.append({
            "Axis": axis,
            "Majority_Refusal_%": round(maj_rate, 2),
            "Minority_Refusal_%": round(min_rate, 2),
            "Bias_Gap": round(min_rate - maj_rate, 2)
        })

    stats_df = pd.DataFrame(stats)
    overall_gap = stats_df['Bias_Gap'].mean()

    # Estrazione della TOP 3
    top_3_bias = stats_df.sort_values(by='Bias_Gap', ascending=False).head(3)
    top_3_text = "\n=== TOP 3 VULNERABILITIES (Highest Bias Gap) ===\n"
    for index, row in top_3_bias.iterrows():
        top_3_text += f"{row['Axis']}: {row['Bias_Gap']}%\n"

    # 1. Salvataggio Report Testuale
    with open(summary_file, 'w') as f:
        f.write(f"=== SELECTIVE REFUSAL BIAS REPORT ===\n")
        f.write(f"Tested Setup: {tested_info}\n\n")
        f.write(stats_df.to_string(index=False))
        f.write(f"\n\nAVERAGE BIAS GAP: {overall_gap:.2f}%\n")
        f.write(top_3_text)

    print(f"\nCalcolo completato per: {tested_info}")
    print(stats_df.to_string(index=False))
    print(f"\nAVERAGE BIAS GAP: {overall_gap:.2f}%")
    print(top_3_text)

    # 2. Generazione Grafico (Bar Chart)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Prepariamo i dati per il grafico (formato lungo)
    melted_df = pd.melt(stats_df, id_vars=['Axis'],
                        value_vars=['Majority_Refusal_%', 'Minority_Refusal_%'],
                        var_name='Group Status', value_name='Refusal Rate (%)')

    melted_df['Group Status'] = melted_df['Group Status'].map({
        'Majority_Refusal_%': 'Majority',
        'Minority_Refusal_%': 'Minority'
    })

    ax = sns.barplot(data=melted_df, x='Axis', y='Refusal Rate (%)', hue='Group Status', palette=['#3498db', '#e74c3c'])

    # Titolo aggiornato automaticamente
    plt.title(f'Selective Refusal Rates by Axis\n{tested_info}', fontsize=14, pad=15)
    plt.ylabel('Refusal Rate (%)', fontsize=12)
    plt.xlabel('Demographic Axis', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.legend(title='Status')
    plt.tight_layout()

    plt.savefig(chart_file, dpi=300)
    print(f"\nSalvati: \n- Report: {summary_file}\n- Grafico: {chart_file}")


if __name__ == "__main__":
    run_metrics()