"""
Dataset loading and management for CatAttack
"""

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from datasets import load_dataset as hf_load_dataset

from .config import DatasetConfig


def load_dataset(config: DatasetConfig) -> List[Dict[str, str]]:
    """Load dataset based on configuration"""
    
    if config.local_path:
        return load_local_dataset(config)
    elif config.name:
        return load_huggingface_dataset(config)
    else:
        raise ValueError("Either dataset name or local path must be specified")


def load_huggingface_dataset(config: DatasetConfig) -> List[Dict[str, str]]:
    """Load dataset from HuggingFace Hub"""
    
    # Load the dataset
    if config.subset:
        dataset = hf_load_dataset(config.name, config.subset, split=config.split)
    else:
        dataset = hf_load_dataset(config.name, split=config.split)
    
    # Convert to list of dictionaries
    problems = []
    for item in dataset:
        problem = {
            config.problem_field: item[config.problem_field],
            config.answer_field: item[config.answer_field]
        }
        problems.append(problem)
    
    # Limit number of problems if specified
    if config.num_problems and config.num_problems < len(problems):
        # Randomly sample problems for diversity
        random.seed(42)  # For reproducibility
        problems = random.sample(problems, config.num_problems)
    
    return problems


def load_local_dataset(config: DatasetConfig) -> List[Dict[str, str]]:
    """Load dataset from local file"""
    
    path = Path(config.local_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    # Determine file format and load accordingly
    if path.suffix.lower() == '.json':
        return load_json_dataset(path, config)
    elif path.suffix.lower() == '.jsonl':
        return load_jsonl_dataset(path, config)
    elif path.suffix.lower() == '.csv':
        return load_csv_dataset(path, config)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_json_dataset(path: Path, config: DatasetConfig) -> List[Dict[str, str]]:
    """Load JSON dataset"""
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        problems = data
    elif isinstance(data, dict) and config.split in data:
        problems = data[config.split]
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")
    
    # Extract relevant fields
    result = []
    for item in problems:
        if config.problem_field in item and config.answer_field in item:
            problem = {
                config.problem_field: item[config.problem_field],
                config.answer_field: item[config.answer_field]
            }
            result.append(problem)
    
    # Limit number of problems
    if config.num_problems and config.num_problems < len(result):
        random.seed(42)
        result = random.sample(result, config.num_problems)
    
    return result


def load_jsonl_dataset(path: Path, config: DatasetConfig) -> List[Dict[str, str]]:
    """Load JSONL dataset"""
    
    problems = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if config.problem_field in item and config.answer_field in item:
                    problem = {
                        config.problem_field: item[config.problem_field],
                        config.answer_field: item[config.answer_field]
                    }
                    problems.append(problem)
    
    # Limit number of problems
    if config.num_problems and config.num_problems < len(problems):
        random.seed(42)
        problems = random.sample(problems, config.num_problems)
    
    return problems


def load_csv_dataset(path: Path, config: DatasetConfig) -> List[Dict[str, str]]:
    """Load CSV dataset"""
    
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load CSV datasets")
    
    df = pd.read_csv(path)
    
    # Check if required columns exist
    if config.problem_field not in df.columns:
        raise ValueError(f"Problem field '{config.problem_field}' not found in CSV")
    if config.answer_field not in df.columns:
        raise ValueError(f"Answer field '{config.answer_field}' not found in CSV")
    
    # Convert to list of dictionaries
    problems = []
    for _, row in df.iterrows():
        problem = {
            config.problem_field: str(row[config.problem_field]),
            config.answer_field: str(row[config.answer_field])
        }
        problems.append(problem)
    
    # Limit number of problems
    if config.num_problems and config.num_problems < len(problems):
        random.seed(42)
        problems = random.sample(problems, config.num_problems)
    
    return problems


def create_sample_dataset(num_problems: int = 10) -> List[Dict[str, str]]:
    """Create a sample dataset for testing"""
    
    sample_problems = [
        {
            "question": "If 5x + 2 = 17, what is x?",
            "answer": "3"
        },
        {
            "question": "A store sells apples for $2 each and oranges for $3 each. If you buy 4 apples and 3 oranges, how much do you spend?",
            "answer": "17"
        },
        {
            "question": "What is 15% of 80?",
            "answer": "12"
        },
        {
            "question": "If a rectangle has length 8 and width 5, what is its area?",
            "answer": "40"
        },
        {
            "question": "Solve for y: 2y - 7 = 11",
            "answer": "9"
        },
        {
            "question": "A car travels 60 miles in 1.5 hours. What is its average speed in miles per hour?",
            "answer": "40"
        },
        {
            "question": "If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads?",
            "answer": "3/8"
        },
        {
            "question": "What is the sum of the first 5 positive integers?",
            "answer": "15"
        },
        {
            "question": "If a triangle has angles of 60° and 70°, what is the third angle?",
            "answer": "50"
        },
        {
            "question": "A box contains 12 red balls and 8 blue balls. What fraction of the balls are red?",
            "answer": "3/5"
        }
    ]
    
    # Return requested number of problems (with repetition if needed)
    if num_problems <= len(sample_problems):
        return sample_problems[:num_problems]
    else:
        # Repeat the list to get enough problems
        multiplier = (num_problems // len(sample_problems)) + 1
        extended = (sample_problems * multiplier)[:num_problems]
        return extended


def save_dataset(problems: List[Dict[str, str]], path: str, format: str = "json") -> None:
    """Save dataset to file"""
    
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "json":
        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
    
    elif format.lower() == "jsonl":
        with open(path_obj, 'w', encoding='utf-8') as f:
            for problem in problems:
                f.write(json.dumps(problem, ensure_ascii=False) + '\n')
    
    elif format.lower() == "csv":
        try:
            import pandas as pd
            df = pd.DataFrame(problems)
            df.to_csv(path_obj, index=False)
        except ImportError:
            raise ImportError("pandas is required to save CSV datasets")
    
    else:
        raise ValueError(f"Unsupported format: {format}")


# Predefined dataset configurations for common datasets
DATASET_CONFIGS = {
    "gsm8k": DatasetConfig(
        name="openai/gsm8k",
        split="test",
        problem_field="question",
        answer_field="answer"
    ),
    
    "math": DatasetConfig(
        name="hendrycks/competition_math",
        split="test",
        problem_field="problem",
        answer_field="solution"
    ),
    
    "numina": DatasetConfig(
        name="AI-MO/NuminaMath-CoT",
        split="train",
        problem_field="problem",
        answer_field="solution"
    ),
    
    "catattack_gsm8k": DatasetConfig(
        name="collinear-ai/catattack_gsm8k",
        split="train",
        problem_field="question",
        answer_field="answer"
    )
}
