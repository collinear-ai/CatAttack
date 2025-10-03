"""
Configuration management for CatAttack
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a model (attacker, proxy_target, target_model, judge, or target)"""
    provider: str  # openai, anthropic, vllm, sglang, bedrock
    model: str
    base_url: Optional[str] = None
    port: Optional[int] = None
    api_key_env: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.0
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from environment variable"""
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


@dataclass
class DatasetConfig:
    """Configuration for dataset"""
    name: Optional[str] = None  # HuggingFace dataset name
    split: str = "test"
    subset: Optional[str] = None
    local_path: Optional[str] = None
    num_problems: int = 100
    problem_field: str = "question"
    answer_field: str = "answer"

    def clone_with_overrides(self, **overrides) -> "DatasetConfig":
        data = self.__dict__.copy()
        data.update(overrides)
        return DatasetConfig(**data)


@dataclass
class TestDatasetConfig(DatasetConfig):
    pass


@dataclass
class AttackConfig:
    """Configuration for attack parameters"""
    max_iterations: int = 10
    num_threads: int = 1


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    log_file: str = "catattack.log"


@dataclass
class OutputConfig:
    """Configuration for output and results"""
    results_dir: str = "results"
    save_triggers: bool = True
    push_to_hub: bool = False
    hub_dataset_name: Optional[str] = None
    hub_private: bool = True
    include_failed_attacks: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for evaluation of suffixes"""
    model_key: str = "target_model"
    num_runs: int = 3
    num_problems: Optional[int] = None
    results_file: str = "evaluation_results.json"


@dataclass
class CatAttackConfig:
    """Main configuration class for CatAttack"""
    models: Dict[str, ModelConfig]
    dataset: DatasetConfig
    test_dataset: TestDatasetConfig
    attack: AttackConfig
    logging: LoggingConfig
    output: OutputConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'CatAttackConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CatAttackConfig':
        """Create configuration from dictionary"""
        # Parse models
        models = {}
        for model_name, model_config in config_dict.get('models', {}).items():
            models[model_name] = ModelConfig(**model_config)
        
        # Parse other sections
        dataset_dict = config_dict.get('dataset', {})
        dataset = DatasetConfig(**dataset_dict)

        test_dataset = TestDatasetConfig(**config_dict.get('test_dataset', {})) if 'test_dataset' in config_dict else TestDatasetConfig(**dataset_dict)
        attack = AttackConfig(**config_dict.get('attack', {}))
        logging = LoggingConfig(**config_dict.get('logging', {}))
        output = OutputConfig(**config_dict.get('output', {}))
        evaluation = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        return cls(
            models=models,
            dataset=dataset,
            test_dataset=test_dataset,
            attack=attack,
            logging=logging,
            output=output,
            evaluation=evaluation
        )
    
    def get_model_config(self, model_type: str) -> ModelConfig:
        """Get configuration for a specific model type"""
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not found in configuration")
        return self.models[model_type]
    
    def validate(self) -> None:
        """Validate the configuration"""
        required_models = ['attacker', 'target_model', 'judge']
        for model_type in required_models:
            if model_type not in self.models:
                raise ValueError(f"Required model '{model_type}' not found in configuration")
        
        # Validate dataset configuration (optional - will use hardcoded samples as fallback)
        # if not self.dataset.name and not self.dataset.local_path:
        #     Warning: Will use hardcoded sample dataset as fallback

        if not self.test_dataset.name and not self.test_dataset.local_path:
            raise ValueError("test_dataset must specify either name or local_path")
        
        # Ensure results directory exists
        Path(self.output.results_dir).mkdir(parents=True, exist_ok=True)

        baseline_dataset_config = self.test_dataset or self.dataset
        if baseline_dataset_config.name is None and baseline_dataset_config.local_path is None:
            raise ValueError("Baseline test dataset requires either name or local_path")


def load_config(config_path: str = "config.yaml") -> CatAttackConfig:
    """Load and validate configuration"""
    config = CatAttackConfig.from_yaml(config_path)
    config.validate()
    return config
