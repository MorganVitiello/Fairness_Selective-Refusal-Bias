import pandas as pd
import argparse
import os
import logging
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run_comparison():
    parser = argparse.ArgumentParser(description="Confronta le valutazioni del Giudice LLM con il Manual Audit.")
    parser.add_argument("--folder", type=str, required=True,
                        help="Percorso della cartella del modello (es. results/bias_granite-3.2-8b_baseline)")
    parser.add_argument("--filename", type=str, default="audit_campione_96.csv",
                        help="Nome del file auditato (default: audit_campione_96.csv)")
    args = parser.parse_args()

    # Percorsi dei file di input
    audited_file = os.path.join(args.folder, args.filename)
    raw_filename = args.filename.replace(".csv", "_raw.csv")
    raw_file = os.path.join(args.folder, "", raw_filename)

    # Percorsi dei file di output (NUOVI)
    errors_filename = args.filename.replace(".csv", "_errors.csv")
    output_errors_file = os.path.join(args.folder, errors_filename)
    report_file = os.path.join(args.folder, "report_metrico.txt")
    plot_file = os.path.join(args.folder, "confusion_matrix.png")

    if not os.path.exists(raw_file):
        logging.error(f"File judge_analisys mancante: {raw_file}")
        return
    if not os.path.exists(audited_file):
        logging.error(f"File auditato mancante: {audited_file}")
        return

    # Caricamento file
    df_raw = pd.read_csv(raw_file)
    df_audited = pd.read_csv(audited_file)

    if len(df_raw) != len(df_audited):
        logging.error(f"Mismatch di righe! Raw: {len(df_raw)} | Audited: {len(df_audited)}.")
        return

    # Estrazione colonne booleane
    raw_eval = df_raw['is_refusal'].astype(bool)
    audited_eval = df_audited['is_refusal'].astype(bool)

    # Calcolo metriche
    total_rows = len(df_raw)
    matches = (raw_eval == audited_eval).sum()
    accuracy = (matches / total_rows) * 100

    false_negatives_mask = (~raw_eval) & audited_eval
    false_negatives = false_negatives_mask.sum()

    false_positives_mask = raw_eval & (~audited_eval)
    false_positives = false_positives_mask.sum()

    true_positives = (raw_eval & audited_eval).sum()
    true_negatives = ((~raw_eval) & (~audited_eval)).sum()

    # Costruzione del testo del Report
    report_text = f"""==================================================
REPORT DI CONFRONTO (LLM vs UMANO)
==================================================
Directory: {args.folder}
File:      {args.filename}
--------------------------------------------------
METRICHE PRINCIPALI:
Righe totali analizzate : {total_rows}
Valutazioni corrette    : {matches} ({accuracy:.2f}%)
Errori totali           : {total_rows - matches} ({(100 - accuracy):.2f}%)
--------------------------------------------------
DETTAGLIO ERRORI:
Falsi Negativi (LLM ha mancato il rifiuto)  : {false_negatives}
Falsi Positivi (LLM ha allucinato un rifiuto): {false_positives}
==================================================
"""

    # 1. Stampa a video e salvataggio nel file TXT
    print("\n" + report_text)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info(f"Report metrico salvato in: {report_file}")

    # 2. Generazione e Salvataggio della Matrice di Confusione
    # L'ordine convenzionale per sklearn è: array reali (umano), array predetti (LLM)
    cm = confusion_matrix(audited_eval, raw_eval, labels=[False, True])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Complied (False)", "Refused (True)"],
                yticklabels=["Complied (False)", "Refused (True)"])
    plt.title(f"Confusion Matrix: LLM Judge vs Human\nAccuracy: {accuracy:.1f}%")
    plt.ylabel("Verità (Human Audit)")
    plt.xlabel("Predizione (LLM Judge)")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)  # Salva in alta risoluzione per la tesi!
    plt.close()
    logging.info(f"Grafico della Matrice di Confusione salvato in: {plot_file}")

    # 3. Salvataggio degli errori (CSV)
    mismatch_mask = raw_eval != audited_eval

    if mismatch_mask.sum() > 0:
        reasoning_col = 'judge_reasoning' if 'judge_reasoning' in df_raw.columns else 'N/A'
        df_errors = pd.DataFrame({
            'prompt': df_audited.loc[mismatch_mask, 'prompt'],
            'response': df_audited.loc[mismatch_mask, 'response'],
            'judge_reasoning': df_raw.loc[mismatch_mask, reasoning_col] if reasoning_col != 'N/A' else "N/A",
            'LLM_Judge_Raw': df_raw.loc[mismatch_mask, 'is_refusal'],
            'Human_Audit': df_audited.loc[mismatch_mask, 'is_refusal']
        })

        df_errors.to_csv(output_errors_file, index=False, sep=';', encoding='utf-8-sig')
        logging.info(f"Salvate le {mismatch_mask.sum()} righe discordanti nel file: {output_errors_file}")


if __name__ == "__main__":
    run_comparison()