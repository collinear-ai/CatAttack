"""Baseline evaluation module for CatAttack."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from src.config import load_config, CatAttackConfig
from src.catattack import CatAttack
from src.dataset import load_dataset
from manual_suffixes import MANUAL_SUFFIXES

logger = logging.getLogger("evaluation")
logging.basicConfig(level=logging.INFO)

@dataclass
class EvaluationResult:
    problem_index: int
    question: str
    ground_truth: str
    baseline_responses: List[str]
    baseline_correctness: List[bool]
    suffix_questions: List[str]
    suffix_correctness: List[bool]
    suffix_responses: List[str]

def get_model_client(catattack: CatAttack, model_key: str):
    if model_key == "attacker":
        return catattack.attacker_client
    if model_key in ("target_model", "proxy_target"):
        return catattack.target_model_client
    if model_key == "judge":
        return catattack.judge_client
    if model_key == "target" and catattack.target_client:
        return catattack.target_client
    raise ValueError(f"Unknown model_key '{model_key}' for evaluation")

async def run_judge(catattack: CatAttack, question: str, answer: str, response: str) -> bool:
    return await catattack.judge_answer(question, answer, response)

async def evaluate_suffixes(config: CatAttackConfig) -> Dict:
    catattack = CatAttack(config)

    test_dataset_config = config.test_dataset
    problems = load_dataset(test_dataset_config)

    if config.evaluation.num_problems:
        problems = problems[:config.evaluation.num_problems]

    model_client = get_model_client(catattack, config.evaluation.model_key)

    evaluation_results: List[EvaluationResult] = []
    total_true = 0
    total_runs = 0

    for idx, problem in enumerate(problems):
        question = problem[test_dataset_config.problem_field]
        ground_truth = problem[test_dataset_config.answer_field]

        baseline_responses: List[str] = []
        baseline_correctness: List[bool] = []

        for _ in range(config.evaluation.num_runs):
            response = await catattack.proxy_client.generate(question)
            baseline_responses.append(response.content)

            is_correct = await run_judge(catattack, question, ground_truth, response.content)
            baseline_correctness.append(is_correct)

            total_runs += 1
            if is_correct:
                total_true += 1

        suffix_questions: List[str] = []
        suffix_correctness: List[bool] = []
        suffix_responses: List[str] = []

        for suffix in MANUAL_SUFFIXES:
            modified_question = f"{question} {suffix}".strip()
            suffix_questions.append(modified_question)

            response = await model_client.generate(modified_question)
            suffix_responses.append(response.content)
            is_correct = await run_judge(catattack, modified_question, ground_truth, response.content)
            suffix_correctness.append(is_correct)

        evaluation_results.append(
            EvaluationResult(
                problem_index=idx,
                question=question,
                ground_truth=ground_truth,
                baseline_responses=baseline_responses,
                baseline_correctness=baseline_correctness,
                suffix_questions=suffix_questions,
                suffix_correctness=suffix_correctness,
                suffix_responses=suffix_responses,
            )
        )

    accuracy = total_true / total_runs if total_runs else 0.0
    error_rate = 1.0 - accuracy

    logger.info("Baseline accuracy: %.2f%%, error rate: %.2f%%", accuracy * 100, error_rate * 100)

    return {
        "summary": {
            "total_problems": len(problems),
            "num_runs": config.evaluation.num_runs,
            "total_evaluations": total_runs,
            "accuracy": accuracy,
            "error_rate": error_rate,
        },
        "results": [
            {
                "problem_index": r.problem_index,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "baseline_responses": r.baseline_responses,
                "baseline_correctness": r.baseline_correctness,
                "suffix_questions": r.suffix_questions,
                "suffix_correctness": r.suffix_correctness,
                "suffix_responses": r.suffix_responses,
            }
            for r in evaluation_results
        ],
    }

async def main(config_path: str = "config.yaml"):
    config = load_config(config_path)

    evaluation_data = await evaluate_suffixes(config)

    output_path = Path(config.output.results_dir) / config.evaluation.results_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2)

    logger.info("Evaluation results saved to %s", output_path)

if __name__ == "__main__":
    asyncio.run(main())
