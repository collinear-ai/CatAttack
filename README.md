# CatAttack: Suffix Trigger Pipeline

CatAttack implements the suffix-attack pipeline described in **Cats Confuse Reasoning LLM: Query-Agnostic Adversarial Triggers for Reasoning Models**. The codebase is organised around two core commands:

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
  attacker:      # attack prompt generation (OpenAI format)
  proxy_target:  # cheaper model used during optimisation loop
  target_model:  # evaluation model for baseline & suffix runs
  judge:         # judge prompt for correctness tests

dataset:         # suffix-generation dataset (attacker loop)
test_dataset:    # evaluation dataset (suffix_evaluator)
attack:          # optimisation settings (iterations, threads)
output:          # results directory & HF push settings
evaluation:      # evaluation runs (num_runs, num_problems, etc.)
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

This script:
- Queries the evaluation model for each question `num_runs` times with the original prompt (baseline).
- Applies every suffix in `MANUAL_SUFFIXES` to each question and re-runs the model.
- Judges all outputs with the configured judge model.

Results are stored in `results/evaluation_results.json` with per-question details plus a summary block.

---

## 4. Metrics Printed (and Stored)

The evaluator prints:

- Baseline accuracy/error rate.
- Per-trigger accuracy/error, average completion tokens, and average token change.
- Growth percentages per trigger: % questions where suffix length ≥1.5×, ≥2×, ≥3×, ≥4× baseline completion tokens.
- Combined suffix accuracy/error (question counts as incorrect if any suffix fails).
- CatAttack ASR (combined error ÷ baseline error), matching the paper’s reporting.
- Overall token statistics (average baseline tokens, average suffix tokens, overall multiplier).

All of these metrics are also written to the `summary` section of `evaluation_results.json` so you can post-process or push them to the Hub if `evaluation.push_to_hub` is enabled.

---

## 5. File Layout

```
src/config.py        # dataclasses & loader
dataset.py           # dataset utilities
models.py            # OpenAI/Anthropic/Fireworks adapters
prompts/             # attacker & judge prompt templates
manual_suffixes.py   # list of human-curated suffixes
suffix_pipeline.py   # optimisation loop
suffix_evaluator.py  # evaluation & metrics
results/             # JSON outputs
```

---

## 6. Citation & Support

If this project helps your research, please cite:

> Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani.  
> **Cats Confuse Reasoning LLM: Query-Agnostic Adversarial Triggers for Reasoning Models**. 2025. [arXiv:2503.01781](https://arxiv.org/abs/2503.01781)

Questions or issues?
- GitHub Issues: https://github.com/collinear-ai/CatAttack/issues
- Email: research@collinear.ai