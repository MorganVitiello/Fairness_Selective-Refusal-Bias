import pandas as pd
from openai import OpenAI
import logging
import os
import argparse
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run_generation():
    parser = argparse.ArgumentParser(description="Generate responses for Bias Analysis.")
    parser.add_argument("--model", type=str, required=True, help="Nome modello in LM Studio (Generatore)")
    parser.add_argument("--phase", type=str, choices=['baseline', 'frontend', 'backend'], required=True)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, "data", "comprehensive_bias_dataset.csv")

    # Crea la directory per i risultati
    result_dir = os.path.join(project_root, "results", f"bias_{args.model}_{args.phase}")
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, "generations.csv")

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    if not os.path.exists(input_file):
        logging.error(f"Dataset non trovato: {input_file}")
        return

    df = pd.read_csv(input_file)
    responses = []

    logging.info(f"Starting Generation Phase | Model: {args.model} | Phase: {args.phase.upper()}")

    for index, row in df.iterrows():
        prompt_text = row['prompt']
        try:
            gen_completion = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0,
                max_tokens=1500
            )
            answer = gen_completion.choices[0].message.content.strip()
            responses.append(answer)
            logging.info(f"[{index + 1}/{len(df)}] Prompt generato.")

        except Exception as e:
            logging.error(f"Errore riga {index + 1}: {e}")
            responses.append(f"ERROR: {e}")

    df['response'] = responses
    df.to_csv(output_file, index=False)
    logging.info(f"Generation completed. Saved raw generations in: {output_file}")


if __name__ == "__main__":
    run_generation()