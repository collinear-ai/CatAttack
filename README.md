# CatAttack

[![arXiv](https://img.shields.io/badge/arXiv-2503.01781-b31b1b.svg)](https://arxiv.org/abs/2503.01781)

**Query-Agnostic Adversarial Triggers for Reasoning Models**

CatAttack discovers universal text suffixes that, when appended to math problems, systematically mislead reasoning models to generate incorrect answers. Based on the paper [*Cats Confuse Reasoning LLM*](https://arxiv.org/abs/2503.01781).

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/collinear-ai/CatAttack.git
cd CatAttack
pip install -e .
```

### 2. Set API Keys

Copy `env.example` to `.env` and fill in your keys:

```bash
cp env.example .env
# Edit .env with your API keys
```

Or export them directly:

```bash
export OPENAI_API_KEY="your-key"
export FIREWORKS_API_KEY="your-key"
```

### 3. Run Suffix Generation

```bash
python -m catattack.cli.suffix_pipeline
```

Generates adversarial suffixes and saves results to `results/catattack_results_*.json`.

### 4. Evaluate Suffixes

Manually review generated suffixes and add the best ones to `src/catattack/manual_suffixes.py`, then run:

```bash
python -m catattack.cli.suffix_evaluator
```

Prints trigger-wise metrics and CatAttack ASR (Attack Success Rate).

---

## Configuration

Edit `config.yaml` to customize your setup. You can also use custom config files:

```bash
python -m catattack.cli.suffix_pipeline my_config.yaml
```

### Complete Configuration Example

```yaml
# Model configurations
models:
  # Attacker: generates adversarial suffix proposals
  attacker:
    provider: "openai"              # openai, anthropic, vllm, sglang
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"  # Environment variable name
    max_tokens: 2048
    temperature: 0.7
    
  # Proxy target: weaker model for fast iteration (used during generation)
  proxy_target:
    provider: "openai"              # Use "openai" for any OpenAI-compatible API
    model: "accounts/fireworks/models/deepseek-v3"
    base_url: "https://api.fireworks.ai/inference/v1"
    api_key_env: "FIREWORKS_API_KEY"
    max_tokens: 4096
    temperature: 0.0
    
  # Target model: stronger model for evaluation (used only in suffix_evaluator)
  target_model:
    provider: "openai"
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"
    max_tokens: 2048
    temperature: 0.0
    
  # Judge: evaluates if answers are correct
  judge:
    provider: "openai"
    model: "gpt-4o-mini"
    api_key_env: "OPENAI_API_KEY"
    max_tokens: 1024
    temperature: 0.0

# Dataset for suffix generation
dataset:
  name: "AI-MO/NuminaMath-CoT"     # HuggingFace dataset (or leave empty for hardcoded samples)
  split: "train"
  num_problems: 100
  problem_field: "problem"          # Field name for questions
  answer_field: "solution"          # Field name for answers
  # local_path: "./problems.json"  # Or use local file (.json, .jsonl, .csv)

# Dataset for suffix evaluation  
test_dataset:
  name: "gsm8k"
  split: "test"
  num_problems: 1000
  problem_field: "question"
  answer_field: "answer"

# Attack parameters
attack:
  max_iterations: 10                # Max attempts per problem to find successful suffix
  num_threads: 2                    # Number of parallel problems to process

# Output settings
output:
  results_dir: "results"
  save_triggers: true
  push_to_hub: false                # Upload generated suffixes to HuggingFace
  hub_dataset_name: "your-org/catattack-problems"  # HF dataset name (if push_to_hub: true)
  hub_private: true
  include_failed_attacks: false     # Include unsuccessful attempts in output

# Evaluation settings
evaluation:
  model_key: "target_model"         # Which model to evaluate on
  num_runs: 6                       # Runs per suffix (for averaging)
  num_problems: 1000                # Number of test problems
  results_file: "evaluation_results.json"  # Saved to results_dir
```

---

## Output

**Suffix Generation** → `results/catattack_results_*.json`
- Original and adversarial questions
- Extracted triggers
- Attack success indicators

**Suffix Evaluation** → `results/evaluation_results.json` + console output
- Baseline accuracy vs suffix accuracy
- Per-trigger performance
- CatAttack ASR (multiplicative increase in error rate)

---

## Citation

```bibtex
@misc{rajeev2025catsconfusereasoningllm,
      title={Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models}, 
      author={Meghana Rajeev and Rajkumar Ramamurthy and Prapti Trivedi and Vikas Yadav and Oluwanifemi Bamgbose and Sathwik Tejaswi Madhusudan and James Zou and Nazneen Rajani},
      year={2025},
      eprint={2503.01781},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.01781}, 
}
```
