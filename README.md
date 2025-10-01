# CatAttack: Query-Agnostic Adversarial Triggers for Reasoning Models

[![arXiv](https://img.shields.io/badge/arXiv-2503.01781-b31b1b.svg)](https://arxiv.org/abs/2503.01781)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repository contains the official implementation of **CatAttack**, an automated method for discovering query-agnostic adversarial triggers for reasoning models, as described in our paper:

> **Cats Confuse Reasoning LLM: Query-Agnostic Adversarial Triggers for Reasoning Models**  
> *Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani*

## 🎯 Overview

CatAttack reveals critical vulnerabilities in state-of-the-art reasoning models by generating short, irrelevant text that, when appended to math problems, systematically misleads models to output incorrect answers without altering the problem's semantics.

### Key Features

- **Automated Attack Generation**: Iterative pipeline for discovering adversarial triggers
- **Transfer Learning**: Attacks discovered on weaker models transfer to stronger reasoning models  
- **Universal Triggers**: Single triggers work across multiple model families
- **Comprehensive Evaluation**: Metrics for attack success rate, response length, and latency impact

### Example Results

Adding simple triggers like *"Interesting fact: cats sleep most of their lives"* to math problems can:
- **Double** the error rate on reasoning models like DeepSeek R1
- Increase error rates by up to **500%** on reasoning models
- Increase error rates by up to **700%** on instruction-tuned models

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/collinear-ai/CatAttack.git
cd CatAttack
pip install -r requirements.txt
```

### Basic Usage

1. **Configure your setup** (copy and modify example config):
```bash
cp examples/config_openai_only.yaml config.yaml
# Edit config.yaml with your API keys and preferences
```

2. **Run CatAttack**:
```bash
python main.py --config config.yaml --sample
```

3. **View results**:
```bash
# Results are saved to results/ directory
cat results/catattack_results_*.json
```

## 📋 Configuration

CatAttack uses YAML configuration files to specify models, datasets, and attack parameters. See the `examples/` directory for sample configurations:

- `config_openai_only.yaml` - Simple setup using only OpenAI models
- `config_local_vllm.yaml` - Advanced setup with local vLLM servers
- `config.yaml` - Full configuration with all options

### Key Configuration Sections

```yaml
models:
  attacker:      # Model that generates adversarial triggers
  proxy_target:  # Weaker model for fast iteration  
  judge:         # Model that evaluates attack success
  target:        # Final evaluation target (reasoning model)

dataset:
  name: "openai/gsm8k"  # HuggingFace dataset
  num_problems: 100     # Number of problems to attack

attack:
  max_iterations: 10    # Max iterations per problem
  max_cost_usd: 50.0   # Budget limit
  trigger_types: ["prefix", "suffix"]
```

## 🔧 Advanced Usage

### Running with Local Models

For research use, we recommend running with local vLLM servers for faster iteration:

```bash
# Start vLLM servers automatically
python main.py --config examples/config_local_vllm.yaml --start-servers

# Or start servers manually
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V3 \
  --port 8000 \
  --tensor-parallel-size 4
```

### Custom Datasets

```python
from src.dataset import load_dataset, DatasetConfig

# Use custom dataset
config = DatasetConfig(
    local_path="my_dataset.json",
    problem_field="question", 
    answer_field="solution"
)
problems = load_dataset(config)
```

### Evaluating Triggers

```python
from src.catattack import CatAttack

catattack = CatAttack(config)
results = await catattack.run_attack()

# Evaluate on transfer models
transfer_results = await catattack.evaluate_triggers(
    results.successful_triggers, 
    test_problems
)
```

## 📊 Results and Analysis

CatAttack generates comprehensive results including:

- **Attack Success Rate**: Percentage of problems where triggers caused incorrect answers
- **Response Length Increase**: How much triggers increase response length  
- **Latency Slowdown**: Impact on model response time
- **Successful Triggers**: List of effective adversarial triggers
- **Transfer Analysis**: How triggers perform across different models

Results are saved in JSON format and can optionally be pushed to HuggingFace Hub.

## 🗂️ Repository Structure

```
CatAttack/
├── src/
│   ├── catattack.py      # Main CatAttack implementation
│   ├── config.py         # Configuration management
│   ├── models.py         # Model clients (OpenAI, vLLM, etc.)
│   ├── dataset.py        # Dataset loading utilities
│   └── utils.py          # Utility functions
├── prompts/
│   ├── attacker.py       # Attacker prompts
│   └── judge.py          # Judge prompts  
├── examples/
│   ├── config_openai_only.yaml
│   └── config_local_vllm.yaml
├── main.py               # CLI entry point
├── config.yaml           # Default configuration
└── requirements.txt
```

## 🎯 Supported Models

### Model Providers
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Anthropic**: Claude-3.5-Sonnet, Claude-3-Haiku  
- **vLLM**: Any HuggingFace model via local server
- **SGLang**: High-performance local inference
- **AWS Bedrock**: Amazon Nova Pro, Claude models

### Tested Reasoning Models
- DeepSeek R1 & R1-distill-qwen-32b
- Qwen QwQ-32B-Preview  
- Microsoft Phi-4
- OpenAI o1 & o3-mini (via API)

## 📚 Citation

If you use CatAttack in your research, please cite our paper:

```bibtex
@article{rajeev2025catattack,
  title={Cats Confuse Reasoning LLM: Query-Agnostic Adversarial Triggers for Reasoning Models},
  author={Rajeev, Meghana and Ramamurthy, Rajkumar and Trivedi, Prapti and Yadav, Vikas and Bamgbose, Oluwanifemi and Madhusudan, Sathwik Tejaswi and Zou, James and Rajani, Nazneen},
  journal={arXiv preprint arXiv:2503.01781},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## ⚠️ Responsible Use

This research is intended to improve the robustness and safety of AI systems. Please use CatAttack responsibly:

- Only test on models you own or have permission to test
- Do not use for malicious purposes
- Report vulnerabilities to model developers through responsible disclosure
- Consider the ethical implications of adversarial testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Support

- **Issues**: [GitHub Issues](https://github.com/collinear-ai/CatAttack/issues)
- **Discussions**: [GitHub Discussions](https://github.com/collinear-ai/CatAttack/discussions)  
- **Email**: research@collinear.ai

## 🔗 Links

- **Paper**: [arXiv:2503.01781](https://arxiv.org/abs/2503.01781)
- **Dataset**: [HuggingFace Hub](https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers)
- **Collinear AI**: [Website](https://collinear.ai)