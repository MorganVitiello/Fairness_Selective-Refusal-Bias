# Selective Refusal Bias in Large Language Models
This repository contains the codebase, datasets, and experimental pipeline for a Thesis at the University of Salerno. The research audits open-weight 8B Large Language Models (LLMs) for the presence of **Selective Refusal Bias** (demographic inconsistencies in safety guardrails) and attempts to mitigate it through black-box prompt engineering. To ensure these mitigations do not degrade overall model helpfulness, the study concludes with a qualitative audit monitoring for **Over-refusal** (false positive refusals on benign prompts).

## Overview
The experiment evaluates four distinct alignment mitigation techniques across two architectural positions (User Prompt vs. System Prompt):
1. **Baseline** (No mitigation)
2. **Persona** (Strictly Neutral Adjudicator)
3. **Chain-of-Thought** (Counterfactual Sanity Check)
4. **Few-Shot** (Out-of-Distribution adversarial examples)

Evaluations are conducted using an automated `LLM-as-a-Judge` architecture to detect refusal triggers and demographic omissions, followed by a Human-in-the-Loop (HITL) qualitative audit to assess the trade-off between helpfulness and safety.


## Repository Structure
* `data/`: Contains the generated datasets.
  * `comprehensive_bias_dataset.csv`: 17 demographic axes for adversarial testing.
  * `qualitative_control_dataset.csv`: Benign prompts for over-refusal auditing.
* `src/`: Source code for the experimental pipeline.
  * `dataset_creation/`: Scripts for programmatic dataset creation.
  * `judge_analysis/`: Utilities for audit comparisons and sample extraction.
  * `generate_responses.py`: Handles JIT (Just-In-Time) loading and stateless generation.
  * `verify_judge.py`: Implements the LLM-as-a-Judge evaluation logic.
  * `calculate_metrics.py`: Computes the Bias Gap for each demographic axis.
* `results/`: Output directories containing raw generations and evaluated CSVs.
* `run_pipeline.py`: The global orchestrator for automated end-to-end execution.


## Prerequisites & Hardware Requirements
To reproduce the experiment locally, the following setup is required:
* **Python**: Version 3.12 or higher.
* **LM Studio**: Configured for local inference, running a local server on port `1234`.
* **Hardware**: A dedicated GPU with at least **~8GB of VRAM** is recommended to smoothly execute Just-In-Time loading and inference for 8B parameter models in GGUF format (e.g., Llama 3.1 8B, Ministral 8B, Granite 3.2 8B).


## Installation
1. Clone the repository:
git clone https://github.com/MorganVitiello/Fairness_Selective-Refusal-Bias.git
cd Fairness_Selective-Refusal-Bias

2. Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate  # On Windows use: .\.venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt


## Usage
The experiment is fully automated via a parameterized CLI. Ensure LM Studio is running before executing the pipeline.

Run the complete end-to-end experiment (Generation, Evaluation, Metrics):
python run_pipeline.py

Run specific experimental blocks:
(The pipeline allows testing specific mitigation techniques or architectural positions) 
For example, to test only the Chain-of-Thought technique in the System Prompt:
python run_pipeline.py --techniques cot --positions system

Available CLI Arguments:
* --models: Specify one or multiple model IDs (must match LM Studio configurations).
* --techniques: baseline, persona, cot, fewshot.
* --positions: user, system.
* --force-data: Force the regeneration of the baseline datasets.
* --skip-gen, --skip-judge, --skip-metrics: Skip specific pipeline phases.


## Key Findings
* **Optimal Mitigation Strategy:** The Few-Shot pattern applied within the User Prompt emerged as the most effective technique. Conversely, injecting instructions into the System Prompt proved significantly less performant and, in some architectures, exacerbated the bias.

* **Quantitative Bias Reduction:** The Few-Shot technique successfully collapsed the initial Selective Refusal Bias gap (which peaked at over 15% in baseline configurations) to under 4% across 3 out of 4 tested models, with two models dropping below the 2% threshold.

* **Absence of Alignment Tax:** The final qualitative audit (HITL) confirmed that this drastic bias reduction did not degrade model usability. Mitigated models retained perfect compliance on benign prompts, proving that demographic symmetry in safety guardrails can be achieved without triggering systemic over-refusal.
