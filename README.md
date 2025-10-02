# CatAttack: Suffix Trigger Pipeline

[![arXiv](https://img.shields.io/badge/arXiv-2503.01781-b31b1b.svg)](https://arxiv.org/abs/2503.01781)

CatAttack implements the suffix-attack pipeline described in **Cats Confuse Reasoning LLM: Query-Agnostic Adversarial Triggers for Reasoning Models** ([arXiv:2503.01781](https://arxiv.org/abs/2503.01781)). The codebase is organised around two core commands:

1. `suffix_pipeline.py` – iteratively generate suffixes with the attacker + proxy loop.
2. `suffix_evaluator.py` – evaluate any suffix list (including human-curated ones) on the target model and print the full metric suite.

---

## 1. Install & Configure

```bash
git clone https://github.com/collinear-ai/CatAttack.git
cd CatAttack
pip install -r requirements.txt
```

Set the environment variables expected in `config.yaml` (e.g. `OPENAI_API_KEY`, `FIREWORKS_API_KEY`). You can copy `.env.example` and run `source .env` before executing scripts.

The main configuration file is `config.yaml`. Key sections:

```yaml
models:
  attacker:      # Model that generates adversarial triggers
  proxy_target:  # Weaker model used for optimisation loop
  target_model:  # Evaluation model (baseline & suffix runs)
  judge:         # Judge used for correctness checking

# Use any dataset with question/answer fields for suffix generation
dataset:
  name: "AI-MO/NuminaMath-CoT"
  split: "test"
  num_problems: 2

# This evaluation dataset is used when running suffix_evaluator.py
test_dataset:
  name: "collinear-ai/TEMP_catattack_codebase"
  split: "train"
  num_problems: 5

attack:
  max_iterations: 10
  num_threads: 2

output:
  results_dir: "results"

```

---

## 2. Generate Suffixes

1. Edit `config.yaml` if you want to change models or datasets.
2. Run the optimisation loop:
   ```bash
   python suffix_pipeline.py
   ```
   - Uses `dataset` and the `proxy_target` model.
   - Saves results to `results/catattack_results_*.json`.
   - Prints discovered suffixes and success rate.
3. **Manually review the generated suffixes** (open the JSON produced in `results/`). Confirm the trigger keeps the problem semantics intact and looks reasonable—mirror the human verification step from the CatAttack paper.
4. Copy any suffixes you want to keep into `manual_suffixes.py`. Those entries will be used by the evaluation stage.

If you want to curate suffixes manually, edit `manual_suffixes.py`. Any strings listed in `MANUAL_SUFFIXES` are evaluated by the next stage.

---

## 3. Evaluate Suffix Impact

1. Configure evaluation parameters in `config.yaml`:
   - `test_dataset`: dataset used for evaluation.
   - `evaluation.model_key`: model evaluated (defaults to `target_model`).
   - `evaluation.num_runs`: baseline runs per question.
   - `evaluation.num_problems`: number of questions to sample.
2. Run the evaluator:
   ```bash
   python suffix_evaluator.py
   ```
   Before running, ensure `evaluation.model_key` points at the model you want to test (by default this is `target_model`), and update the `target_model` block in `config.yaml` if you want to evaluate a different provider/model.

This script:
- Queries the evaluation model for each question `num_runs` times with the original prompt (baseline).
- Applies every suffix in `MANUAL_SUFFIXES` to each question and re-runs the model.
- Judges all outputs with the configured judge model.