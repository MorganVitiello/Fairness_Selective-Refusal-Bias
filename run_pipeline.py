import os
import subprocess
import time
import argparse

# ==========================================
# CONFIGURAZIONE PIPELINE GLOBALE & PATH
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Modelli di default se non specificati
DEFAULT_MODELS = [
    "ministral-3-8b-instruct-2512",
    "ibm/granite-3.2-8b",
    "gemma-4-e4b-it",
    "meta-llama-3.1-8b-instruct"
]
JUDGE_MODEL = "meta-llama-3.1-8b-instruct"

# Nome del file di output atteso per il dataset principale
MAIN_DATASET_FILE = os.path.join(DATA_DIR, "comprehensive_bias_dataset.csv")
QUAL_DATASET_FILE = os.path.join(DATA_DIR, "qualitative_control_dataset.csv")


def run_cmd(cmd):
    """Esegue un comando a terminale dalla root del progetto."""
    print(f"\n[Esecuzione] > {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=BASE_DIR)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRORE CRITICO] Il comando ha fallito: {cmd}")
        exit(1)


def main():
    # --- SETUP ARGOMENTI DA TERMINALE ---
    parser = argparse.ArgumentParser(description="Pipeline End-to-End per LLM Fairness Audit")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Lista dei modelli da testare")
    parser.add_argument("--techniques", nargs="+", default=["persona", "cot", "fewshot"], help="Tecniche da testare")
    parser.add_argument("--positions", nargs="+", default=["user", "system"], help="Posizionamenti da testare")
    parser.add_argument("--force-data", action="store_true", help="Forza la rigenerazione del dataset anche se esiste")
    parser.add_argument("--skip-gen", action="store_true", help="Salta la Fase 1 (Generazione delle risposte)")
    parser.add_argument("--skip-judge", action="store_true", help="Salta la Fase 2 (Valutazione automatica)")
    parser.add_argument("--skip-metrics", action="store_true", help="Salta la Fase 3 (Calcolo delle metriche)")

    args = parser.parse_args()

    print("=" * 60)
    print(" PIPELINE SPERIMENTALE END-TO-END (JIT LOADING ATTIVO)")
    print(f" Modelli configurati: {args.models}")
    print(f" Tecniche configurate: {args.techniques}")
    print(f" Posizionamenti configurati: {args.positions}")
    print("=" * 60)

    # ---------------------------------------------------------
    # FASE 0: CONTROLLO/GENERAZIONE DATASET
    # ---------------------------------------------------------
    print("\n\n" + "=" * 40 + "\n FASE 0: CONTROLLO DATASET INIZIALI\n" + "=" * 40)

    # 1. Dataset Principale (Adversarial)
    if not os.path.exists(MAIN_DATASET_FILE) or args.force_data:
        print("[INFO] Dataset principale non trovato o rigenerazione forzata. Avvio la creazione...")
        cmd_gen_data = f"python {os.path.join(SRC_DIR, 'dataset_creation', 'generate_db.py')}"
        run_cmd(cmd_gen_data)
    else:
        print(f"[INFO] Dataset principale rilevato in: {MAIN_DATASET_FILE}")

    # 2. Dataset Qualitativo (Benign)
    if not os.path.exists(QUAL_DATASET_FILE) or args.force_data:
        print("[INFO] Dataset qualitativo non trovato o rigenerazione forzata. Avvio la creazione...")
        cmd_gen_qual = f"python {os.path.join(SRC_DIR, 'dataset_creation', 'qualitative_db.py')}"
        run_cmd(cmd_gen_qual)
    else:
        print(f"[INFO] Dataset qualitativo rilevato in: {QUAL_DATASET_FILE}")

    # ---------------------------------------------------------
    # FASE 1: GENERAZIONE RISPOSTE
    # ---------------------------------------------------------
    if not args.skip_gen:
        print("\n\n" + "=" * 40 + "\n FASE 1: GENERAZIONE RISPOSTE\n" + "=" * 40)
        for model in args.models:
            print(f"\n[JIT LOADING] Attesa di 15s per lo svuotamento della VRAM prima di {model}...")
            time.sleep(15)

            for tech in args.techniques:
                for pos in args.positions:
                    cmd_gen = f"python {os.path.join(SRC_DIR, 'generate_responses.py')} --model \"{model}\" --technique {tech} --position {pos}"
                    run_cmd(cmd_gen)
    else:
        print("\n[SKIP] Fase 1 (Generazione) saltata dall'utente.")

    # ---------------------------------------------------------
    # FASE 2: LLM-AS-A-JUDGE
    # ---------------------------------------------------------
    if not args.skip_judge:
        print("\n\n" + "=" * 40 + "\n FASE 2: VALUTAZIONE GIUDICE\n" + "=" * 40)
        print(f"\n[JIT LOADING] Caricamento automatico del Giudice: {JUDGE_MODEL}")
        time.sleep(15)

        for model in args.models:
            for tech in args.techniques:
                for pos in args.positions:
                    gen_file = os.path.join(RESULTS_DIR, pos, tech, model, "generations.csv")
                    eval_file = os.path.join(RESULTS_DIR, pos, tech, model, "bias_results.csv")

                    if os.path.exists(gen_file):
                        run_cmd(
                            f"python {os.path.join(SRC_DIR, 'verify_judge.py')} --input \"{gen_file}\" --output \"{eval_file}\"")
                    else:
                        print(f"[AVVISO] File generato non trovato. Salto la valutazione per: {gen_file}")
    else:
        print("\n[SKIP] Fase 2 (Giudice) saltata dall'utente.")

    # ---------------------------------------------------------
    # FASE 3: METRICHE
    # ---------------------------------------------------------
    if not args.skip_metrics:
        print("\n\n" + "=" * 40 + "\n FASE 3: CALCOLO METRICHE (BIAS GAP)\n" + "=" * 40)
        for model in args.models:
            for tech in args.techniques:
                for pos in args.positions:
                    eval_file = os.path.join(RESULTS_DIR, pos, tech, model, "bias_results.csv")
                    if os.path.exists(eval_file):
                        run_cmd(f"python {os.path.join(SRC_DIR, 'calculate_metrics.py')} --input_csv \"{eval_file}\"")
                    else:
                        print(f"[AVVISO] File valutato non trovato. Salto le metriche per: {eval_file}")
    else:
        print("\n[SKIP] Fase 3 (Metriche) saltata dall'utente.")

    print("\n" + "=" * 60 + "\n PIPELINE COMPLETATA CON SUCCESSO! \n" + "=" * 60)


if __name__ == "__main__":
    main()