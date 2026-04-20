import os
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def prepare_dataset():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    data_dir = os.path.join(project_root, 'data')
    env_path = os.path.join(project_root, '.env')

    load_dotenv(dotenv_path=env_path)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logging.info(f"Cartella creata in: {data_dir}")

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
        'Facial Scars', 'Schizophrenia Symptomatic', 'Speech Disability',
        'Blind Completely', 'Mental Retardation', 'HIV Positive', 'Obesity',
        'Deaf Completely', 'Quadriplegia', 'Multiple Sclerosis',
        'Cancer Survivor', 'Down Syndrome', 'Autism Spectrum',
        'Alcoholism', 'Homelessness', 'Criminal Record', 'Drug Addiction Recovery',
        'Unemployment', 'Illiteracy', 'Ex-Convict', 'Single Parent',
        'Transgender', 'Non-binary', 'Refugee Status', 'Undocumented Immigrant',
        'Atheism', 'Religious Minority', 'Elderly 80+', 'Neurodivergent', 'Low Income',
        'Depression', 'Bipolar Disorder', 'Borderline Personality Disorder',
        'Anorexia Nervosa', 'Tourette Syndrome', 'Burn Victim', 'Amputee',
        'Dwarfism', 'Wheelchair User', 'Cerebral Palsy', 'Gay', 'Lesbian',
        'Bisexual', 'Sex Worker', 'Polyamorous',
        'Muslim', 'Jewish', 'Mormon', 'Scientologist', 'Asexual', 'Pansexual',
        'Epilepsy', 'Parkinson\'s Disease', 'Psoriasis', 'Albinism', 'Leprosy',
        'Obsessive Compulsive Disorder', 'Post Traumatic Stress Disorder',
        'Domestic Violence Survivor', 'Teenage Parent', 'Welfare Recipient',
        'Child of Incarcerated Parent', 'Illegal Alien', 'Adoption', 'Foster Care'
    ]

    stigma_col = 'stigma' if 'stigma' in df_filtered.columns else 'stigma_type'
    df_final = df_filtered[df_filtered[stigma_col].isin(target_stigmas)]

    # 5. SALVATAGGIO
    output_filename = os.path.join(data_dir, "dataset_filtrato_esperimento.csv")
    df_final.to_csv(output_filename, index=False)

    logging.info("=== Pipeline Completata! ===")
    logging.info(f"File salvato come: '{output_filename}'")
    logging.info(f"Totale righe pronte per l'esperimento: {len(df_final)}")


if __name__ == "__main__":
    prepare_dataset()