import os

from datasets import load_dataset
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def prepare_dataset():
    if not os.path.exists('data'):
        os.makedirs('data')

    logging.info("Avvio il download del dataset SocialStigmaQA da Hugging Face...")

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("ERRORE: HF_TOKEN non trovato nel file .env")
        return


    try:
        # 1. Caricamento e autenticazione
        dataset_dict = load_dataset(
            "ibm-research/SocialStigmaQA",
            "yes_no",
            token=hf_token
        )

        # 2. Estrazione dinamica dello split
        split_name = list(dataset_dict.keys())[0]
        logging.info(f"Trovata partizione '{split_name}'. Conversione in Pandas...")
        df = dataset_dict[split_name].to_pandas()

        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        logging.info(f"Colonne normalizzate: {list(df.columns)}")

    except Exception as e:
        logging.error(f"Errore nel download. Hai accettato i termini sul sito e inserito il token valido?")
        logging.error(f"Dettagli tecnici: {e}")
        return

    logging.info(f"Dataset scaricato. Totale righe iniziali: {len(df)}")

    # 3. FILTRAGGIO STILE: Teniamo solo il prompt 'original'
    if 'prompt_style' in df.columns:
        df_filtered = df[df['prompt_style'] == 'original'].copy()
    else:
        logging.error(f"Colonna 'prompt_style' non trovata! Colonne presenti: {list(df.columns)}")
        return

    # 4. SELEZIONE STIGMI
    target_stigmas = [
        # Disabilità e Condizioni Mediche
        'Facial Scars', 'Schizophrenia Symptomatic', 'Speech Disability',
        'Blind Completely', 'Mental Retardation', 'HIV Positive', 'Obesity',
        'Deaf Completely', 'Quadriplegia', 'Multiple Sclerosis',
        'Cancer Survivor', 'Down Syndrome', 'Autism Spectrum',

        # Condizioni Sociali e Comportamentali
        'Alcoholism', 'Homelessness', 'Criminal Record', 'Drug Addiction Recovery',
        'Unemployment', 'Illiteracy', 'Ex-Convict', 'Single Parent',

        # Identità e Background
        'Transgender', 'Non-binary', 'Refugee Status', 'Undocumented Immigrant',
        'Atheism', 'Religious Minority', 'Elderly 80+', 'Neurodivergent', 'Low Income'
    ]

    stigma_col = 'stigma' if 'stigma' in df_filtered.columns else 'stigma_type'
    df_final = df_filtered[df_filtered[stigma_col].isin(target_stigmas)]

    # 5. SALVATAGGIO
    output_path = os.path.join("data", "dataset_filtrato_esperimento.csv")
    df_final.to_csv(output_path, index=False)

    logging.info("=== Pipeline Completata! ===")
    logging.info(f"File salvato come: '{output_path}'")
    logging.info(f"Totale righe pronte per l'esperimento: {len(df_final)}")


if __name__ == "__main__":
    prepare_dataset()