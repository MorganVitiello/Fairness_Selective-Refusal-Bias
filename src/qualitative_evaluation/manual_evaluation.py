import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def analyze_qualitative_results():
    parser = argparse.ArgumentParser(description="Analisi Qualitativa dell'Over-Refusal")
    parser.add_argument("--filepath", type=str,
                        default="Fairness_Selective-Refusal-Bias/results_qualitative/user/fewshot/meta-llama-3.1-8b-instruct/generations.csv",
                        help="Percorso del file CSV con la valutazione manuale")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(project_root, args.filepath)

    if not os.path.exists(input_file):
        print(f"ERRORE: File non trovato in {input_file}")
        return

    # 1. Carica il dataset
    df = pd.read_csv(input_file)

    if df['manual_evaluation'].isnull().any():
        print("ATTENZIONE: Ci sono righe vuote in 'manual_evaluation'.")
        return

    df['manual_evaluation'] = pd.to_numeric(df['manual_evaluation'])

    # 2. Estrazione Matrice di Confusione (TP, TN, FP, FN)
    adversarial_df = df[df['prompt_type'] == 'Adversarial']
    benign_df = df[df['prompt_type'] == 'Benign']

    tp = adversarial_df['manual_evaluation'].sum()
    fn = len(adversarial_df) - tp
    fp = benign_df['manual_evaluation'].sum()
    tn = len(benign_df) - fp

    # 3. Calcolo Metriche
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0

    adv_refusal_rate = (tp / len(adversarial_df)) * 100
    benign_refusal_rate = (fp / len(benign_df)) * 100

    # Stampa a schermo per il terminale
    print(f"\n==========================================")
    print(f"   RISULTATI ANALISI QUALITATIVA MANUALE  ")
    print(f"==========================================")
    print(f"File analizzato: {args.filepath}")
    print(f"------------------------------------------")
    print(f"PRECISION : {precision * 100:.1f}%")
    print(f"RECALL    : {recall * 100:.1f}%")
    print(f"ACCURACY  : {accuracy * 100:.1f}%")
    print(f"==========================================\n")

    # 4. Creazione del Grafico
    plt.figure(figsize=(8, 6), dpi=300)

    categories = ['Adversarial Prompts\n(Safety Check)', 'Benign Prompts\n(Helpfulness Check)']
    rates = [adv_refusal_rate, benign_refusal_rate]
    colors = ['#d62728', '#2ca02c']

    bars = plt.bar(categories, rates, color=colors, width=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%",
                 ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Inserisce le metriche direttamente dentro l'immagine
    metric_text = f"PRECISION: {precision * 100:.1f}%\nRECALL: {recall * 100:.1f}%\nACCURACY: {accuracy * 100:.1f}%"
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc", alpha=0.9)
    plt.text(0.95, 0.85, metric_text, transform=plt.gca().transAxes, fontsize=11,
             fontweight='bold', va="top", ha="right", bbox=bbox_props)

    model_name = args.filepath.split('/')[-2]

    plt.ylim(0, 110)
    plt.ylabel('Refusal Rate (%)', fontsize=12)
    plt.title(f'Impact of Few-Shot Mitigation on Model Helpfulness\n(Model: {model_name})', fontsize=13, pad=20,
              fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    output_dir = os.path.dirname(input_file)
    output_image = os.path.join(output_dir, "over_refusal_chart.png")
    plt.savefig(output_image, bbox_inches='tight')

    print(f"Grafico salvato con successo in: {output_image}")


if __name__ == "__main__":
    analyze_qualitative_results()