import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def run_lecturing_metrics():
    parser = argparse.ArgumentParser(description="Calculate Lecturing Bias metrics and generate charts.")
    parser.add_argument("--input_csv", type=str, required=True, help="Percorso al file _with_lecturing.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Errore: File non trovato {args.input_csv}")
        return

    result_dir = os.path.dirname(args.input_csv)
    base_filename = os.path.splitext(os.path.basename(args.input_csv))[0]
    # Pulisce il nome per il report
    tested_info = base_filename.replace("_with_lecturing", "").replace("risultati_bias_", "")

    summary_file = os.path.join(result_dir, f"summary_lecturing_{tested_info}.txt")
    chart_file = os.path.join(result_dir, f"chart_lecturing_{tested_info}.png")

    df = pd.read_csv(args.input_csv)

    # Assicuriamo i tipi di dati
    df['is_refusal'] = pd.to_numeric(df['is_refusal'], errors='coerce').fillna(False).astype(bool)
    if 'has_lecturing' not in df.columns:
        print("Errore: La colonna 'has_lecturing' non è presente nel CSV.")
        return
    df['has_lecturing'] = pd.to_numeric(df['has_lecturing'], errors='coerce').fillna(False).astype(bool)

    stats = []
    # Calcoliamo le metriche SOLO sulle righe in cui il modello ha ubbidito (is_refusal == False)
    compliant_df = df[df['is_refusal'] == False]

    for axis in compliant_df['axis'].unique():
        sub = compliant_df[compliant_df['axis'] == axis]

        maj_sub = sub[sub['status'] == 'Majority']['has_lecturing']
        min_sub = sub[sub['status'] == 'Minority']['has_lecturing']

        maj_rate = maj_sub.mean() * 100 if len(maj_sub) > 0 else 0
        min_rate = min_sub.mean() * 100 if len(min_sub) > 0 else 0

        stats.append({
            "Axis": axis,
            "Majority_Lecturing_%": round(maj_rate, 2),
            "Minority_Lecturing_%": round(min_rate, 2),
            "Lecturing_Gap": round(min_rate - maj_rate, 2)
        })

    stats_df = pd.DataFrame(stats)
    overall_gap = stats_df['Lecturing_Gap'].mean()

    # --- NUOVO: Estrazione della TOP 3 degli assi con più paternali asimmetriche ---
    top_3_lecturing = stats_df.sort_values(by='Lecturing_Gap', ascending=False).head(3)
    top_3_text = "\n=== TOP 3 LECTURING ASYMMETRIES (Highest Gap) ===\n"
    for index, row in top_3_lecturing.iterrows():
        top_3_text += f"{row['Axis']}: {row['Lecturing_Gap']}%\n"

    # 1. Salvataggio Report Testuale
    with open(summary_file, 'w') as f:
        f.write(f"=== LECTURING BIAS REPORT (Reluctant Compliance) ===\n")
        f.write(f"Tested Setup: {tested_info}\n")
        f.write("NOTE: Percentages calculated ONLY on prompts where the AI complied (is_refusal = False).\n\n")
        f.write(stats_df.to_string(index=False))
        f.write(f"\n\nAVERAGE LECTURING GAP: {overall_gap:.2f}%\n")
        f.write(top_3_text)

    print(f"\nCalcolo completato per: {tested_info}")
    print(stats_df.to_string(index=False))
    print(f"\nAVERAGE LECTURING GAP: {overall_gap:.2f}%")
    print(top_3_text)

    # 2. Generazione Grafico (Bar Chart)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    melted_df = pd.melt(stats_df, id_vars=['Axis'],
                        value_vars=['Majority_Lecturing_%', 'Minority_Lecturing_%'],
                        var_name='Group Status', value_name='Lecturing Rate (%)')

    melted_df['Group Status'] = melted_df['Group Status'].map({
        'Majority_Lecturing_%': 'Majority',
        'Minority_Lecturing_%': 'Minority'
    })

    # Usiamo colori diversi dal Bias Gap (es. Verde e Viola) per distinguerli visivamente nella tesi
    ax = sns.barplot(data=melted_df, x='Axis', y='Lecturing Rate (%)', hue='Group Status',
                     palette=['#2ecc71', '#9b59b6'])
    plt.title(f'Lecturing Rates (Reluctant Compliance) by Axis\n({tested_info})', fontsize=14, pad=15)
    plt.ylabel('Lecturing Rate (%)', fontsize=12)
    plt.xlabel('Demographic Axis', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.legend(title='Status')
    plt.tight_layout()

    plt.savefig(chart_file, dpi=300)
    print(f"\nSalvati: \n- Report: {summary_file}\n- Grafico: {chart_file}")


if __name__ == "__main__":
    run_lecturing_metrics()