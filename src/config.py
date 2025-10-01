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
    """Configuration for a model (attacker, proxy_target, judge, or target)"""
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
class CatAttackConfig:
    """Main configuration class for CatAttack"""
    models: Dict[str, ModelConfig]
    dataset: DatasetConfig
    attack: AttackConfig
    logging: LoggingConfig
    output: OutputConfig
    
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
        dataset = DatasetConfig(**config_dict.get('dataset', {}))
        attack = AttackConfig(**config_dict.get('attack', {}))
        logging = LoggingConfig(**config_dict.get('logging', {}))
        output = OutputConfig(**config_dict.get('output', {}))
        
        return cls(
            models=models,
            dataset=dataset,
            attack=attack,
            logging=logging,
            output=output
        )
    
    def get_model_config(self, model_type: str) -> ModelConfig:
        """Get configuration for a specific model type"""
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not found in configuration")
        return self.models[model_type]
    
    def validate(self) -> None:
        """Validate the configuration"""
        required_models = ['attacker', 'proxy_target', 'judge']
        for model_type in required_models:
            if model_type not in self.models:
                raise ValueError(f"Required model '{model_type}' not found in configuration")
        
        # Validate dataset configuration
        if not self.dataset.name and not self.dataset.local_path:
            raise ValueError("Either dataset.name or dataset.local_path must be specified")
        
        # Create output directory if it doesn't exist
        Path(self.output.results_dir).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = "config.yaml") -> CatAttackConfig:
    """Load and validate configuration"""
    config = CatAttackConfig.from_yaml(config_path)
    config.validate()
    return config
