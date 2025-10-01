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
    baseline_completion_tokens: List[int]
    suffix_questions: List[str]
    suffix_correctness: List[bool]
    suffix_responses: List[str]
    suffix_completion_tokens: List[int]
    suffix_token_multipliers: List[float]

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
    baseline_client = model_client

    evaluation_results: List[EvaluationResult] = []
    total_true = 0
    total_runs = 0

    for idx, problem in enumerate(problems):
        question = problem[test_dataset_config.problem_field]
        ground_truth = problem[test_dataset_config.answer_field]

        baseline_responses: List[str] = []
        baseline_correctness: List[bool] = []
        baseline_tokens: List[int] = []

        for _ in range(config.evaluation.num_runs):
            response = await baseline_client.generate(question)
            baseline_responses.append(response.content)
            baseline_tokens.append(response.completion_tokens)

            is_correct = await run_judge(catattack, question, ground_truth, response.content)
            baseline_correctness.append(is_correct)

            total_runs += 1
            if is_correct:
                total_true += 1

        suffix_questions: List[str] = []
        suffix_correctness: List[bool] = []
        suffix_responses: List[str] = []
        suffix_tokens: List[int] = []
        suffix_token_multipliers: List[float] = []

        avg_baseline_tokens = (sum(baseline_tokens) / len(baseline_tokens)) if baseline_tokens else 1.0

        for suffix in MANUAL_SUFFIXES:
            modified_question = f"{question} {suffix}".strip()
            suffix_questions.append(modified_question)

            response = await model_client.generate(modified_question)
            suffix_responses.append(response.content)
            suffix_tokens.append(response.completion_tokens)
            multiplier = (response.completion_tokens / avg_baseline_tokens) if avg_baseline_tokens else 0.0
            suffix_token_multipliers.append(multiplier)

            is_correct = await run_judge(catattack, modified_question, ground_truth, response.content)
            suffix_correctness.append(is_correct)

        evaluation_results.append(
            EvaluationResult(
                problem_index=idx,
                question=question,
                ground_truth=ground_truth,
                baseline_responses=baseline_responses,
                baseline_correctness=baseline_correctness,
                baseline_completion_tokens=baseline_tokens,
                suffix_questions=suffix_questions,
                suffix_correctness=suffix_correctness,
                suffix_responses=suffix_responses,
                suffix_completion_tokens=suffix_tokens,
                suffix_token_multipliers=suffix_token_multipliers,
            )
        )

    baseline_accuracy = total_true / total_runs if total_runs else 0.0
    baseline_error_rate = 1.0 - baseline_accuracy

    num_questions = len(evaluation_results)
    num_suffixes = len(MANUAL_SUFFIXES)
    suffix_accuracy_per_trigger: List[float] = []
    suffix_error_per_trigger: List[float] = []

    if num_suffixes and num_questions:
        for idx in range(num_suffixes):
            true_count = sum(1 for r in evaluation_results if len(r.suffix_correctness) > idx and r.suffix_correctness[idx])
            acc = true_count / num_questions
            suffix_accuracy_per_trigger.append(acc)
            suffix_error_per_trigger.append(1.0 - acc)

    combined_correct = sum(1 for r in evaluation_results if r.suffix_correctness and all(r.suffix_correctness))
    if num_suffixes == 0:
        combined_accuracy = 1.0
    else:
        combined_accuracy = combined_correct / num_questions if num_questions else 0.0
    combined_error_rate = 1.0 - combined_accuracy

    if baseline_error_rate > 0:
        catattack_asr = combined_error_rate / baseline_error_rate
    else:
        catattack_asr = None

    all_baseline_tokens = [token for r in evaluation_results for token in r.baseline_completion_tokens]
    avg_baseline_tokens = (sum(all_baseline_tokens) / len(all_baseline_tokens)) if all_baseline_tokens else 0.0

    suffix_tokens_per_trigger: List[List[int]] = [[] for _ in range(num_suffixes)]
    all_suffix_tokens: List[int] = []
    for r in evaluation_results:
        for idx, token in enumerate(r.suffix_completion_tokens):
            if idx < num_suffixes:
                suffix_tokens_per_trigger[idx].append(token)
        all_suffix_tokens.extend(r.suffix_completion_tokens)

    suffix_avg_completion_tokens_per_trigger: List[float] = []
    suffix_avg_length_change_per_trigger: List[float] = []
    suffix_growth_counts_per_trigger = [[0, 0, 0, 0] for _ in range(num_suffixes)]

    if num_suffixes:
        for idx in range(num_suffixes):
            values = suffix_tokens_per_trigger[idx]
            avg_tokens = (sum(values) / len(values)) if values else 0.0
            suffix_avg_completion_tokens_per_trigger.append(avg_tokens)
            suffix_avg_length_change_per_trigger.append(avg_tokens - avg_baseline_tokens)

        for r in evaluation_results:
            for idx, multiplier in enumerate(r.suffix_token_multipliers):
                if idx < num_suffixes:
                    if multiplier >= 1.5:
                        suffix_growth_counts_per_trigger[idx][0] += 1
                    if multiplier >= 2.0:
                        suffix_growth_counts_per_trigger[idx][1] += 1
                    if multiplier >= 3.0:
                        suffix_growth_counts_per_trigger[idx][2] += 1
                    if multiplier >= 4.0:
                        suffix_growth_counts_per_trigger[idx][3] += 1

    avg_suffix_tokens = (sum(all_suffix_tokens) / len(all_suffix_tokens)) if all_suffix_tokens else 0.0
    avg_suffix_length_change = avg_suffix_tokens - avg_baseline_tokens
    overall_multiplier = (avg_suffix_tokens / avg_baseline_tokens) if avg_baseline_tokens else None

    return {
        "summary": {
            "total_problems": len(problems),
            "num_runs": config.evaluation.num_runs,
            "total_evaluations": total_runs,
            "baseline_accuracy": baseline_accuracy,
            "baseline_error_rate": baseline_error_rate,
            "suffix_accuracy_per_trigger": suffix_accuracy_per_trigger,
            "suffix_error_rate_per_trigger": suffix_error_per_trigger,
            "combined_suffix_accuracy": combined_accuracy,
            "combined_suffix_error_rate": combined_error_rate,
            "catattack_asr": catattack_asr,
            "avg_baseline_completion_tokens": avg_baseline_tokens,
            "avg_suffix_completion_tokens": avg_suffix_tokens,
            "avg_suffix_length_change": avg_suffix_length_change,
            "suffix_avg_completion_tokens_per_trigger": suffix_avg_completion_tokens_per_trigger,
            "suffix_avg_length_change_per_trigger": suffix_avg_length_change_per_trigger,
            "suffix_token_growth_counts_per_trigger": [
                {
                    ">=1.5x": (counts[0] / num_questions) if num_questions else 0.0,
                    ">=2x": (counts[1] / num_questions) if num_questions else 0.0,
                    ">=3x": (counts[2] / num_questions) if num_questions else 0.0,
                    ">=4x": (counts[3] / num_questions) if num_questions else 0.0,
                }
                for counts in suffix_growth_counts_per_trigger
            ],
            "overall_suffix_token_multiplier": overall_multiplier,
        },
        "results": [
            {
                "problem_index": r.problem_index,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "baseline_responses": r.baseline_responses,
                "baseline_correctness": r.baseline_correctness,
                "baseline_completion_tokens": r.baseline_completion_tokens,
                "avg_baseline_completion_tokens": (sum(r.baseline_completion_tokens) / len(r.baseline_completion_tokens)) if r.baseline_completion_tokens else 0.0,
                "suffix_questions": r.suffix_questions,
                "suffix_responses": r.suffix_responses,
                "suffix_correctness": r.suffix_correctness,
                "suffix_completion_tokens": r.suffix_completion_tokens,
                "avg_suffix_completion_tokens": (sum(r.suffix_completion_tokens) / len(r.suffix_completion_tokens)) if r.suffix_completion_tokens else 0.0,
                "suffix_token_multipliers": r.suffix_token_multipliers,
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

    summary = evaluation_data["summary"]
    baseline_accuracy = summary.get("baseline_accuracy", 0.0)
    baseline_error_rate = summary.get("baseline_error_rate", 0.0)
    suffix_accuracy = summary.get("suffix_accuracy_per_trigger", [])
    suffix_error = summary.get("suffix_error_rate_per_trigger", [])
    combined_accuracy = summary.get("combined_suffix_accuracy", 0.0)
    combined_error = summary.get("combined_suffix_error_rate", 0.0)
    catattack_asr = summary.get("catattack_asr")
    avg_baseline_tokens = summary.get("avg_baseline_completion_tokens", 0.0)
    avg_suffix_tokens = summary.get("avg_suffix_completion_tokens", 0.0)
    avg_suffix_length_change = summary.get("avg_suffix_length_change", 0.0)
    suffix_avg_tokens_per_trigger = summary.get("suffix_avg_completion_tokens_per_trigger", [])
    suffix_avg_length_change_per_trigger = summary.get("suffix_avg_length_change_per_trigger", [])
    suffix_growth_counts_per_trigger = summary.get("suffix_token_growth_counts_per_trigger", [])
    overall_multiplier = summary.get("overall_suffix_token_multiplier")

    print("\n=== Evaluation Metrics ===")
    print(f"Baseline accuracy: {baseline_accuracy:.2%} (error rate {baseline_error_rate:.2%})")

    if MANUAL_SUFFIXES:
        print("\nSuffix Accuracy by Trigger:")
        for idx, (acc, err) in enumerate(zip(suffix_accuracy, suffix_error), start=1):
            suffix_preview = MANUAL_SUFFIXES[idx - 1][:60] + ("..." if len(MANUAL_SUFFIXES[idx - 1]) > 60 else "")
            length_change = suffix_avg_length_change_per_trigger[idx - 1] if idx - 1 < len(suffix_avg_length_change_per_trigger) else 0.0
            avg_tokens = suffix_avg_tokens_per_trigger[idx - 1] if idx - 1 < len(suffix_avg_tokens_per_trigger) else 0.0
            growth = suffix_growth_counts_per_trigger[idx - 1] if idx - 1 < len(suffix_growth_counts_per_trigger) else {">=1.5x":0, ">=2x":0, ">=3x":0, ">=4x":0}
            print(f"Trigger {idx}: '{suffix_preview}' -> accuracy {acc:.2%}, error {err:.2%}, avg tokens {avg_tokens:.2f}, Δtokens {length_change:+.2f}")
            print(f"  Growth ≥1.5x: {growth.get('>=1.5x',0):.2%}, ≥2x: {growth.get('>=2x',0):.2%}, ≥3x: {growth.get('>=3x',0):.2%}, ≥4x: {growth.get('>=4x',0):.2%}")

    print(f"\nCombined suffix accuracy: {combined_accuracy:.2%}")
    print(f"Combined suffix error rate: {combined_error:.2%}")
    if catattack_asr is not None and baseline_error_rate > 0:
        print(f"CatAttack ASR (combined error / baseline error): {catattack_asr:.2f}")
    else:
        print("CatAttack ASR: undefined (baseline error rate is zero)")

    print("\nCompletion Token Summary:")
    print(f"Average baseline completion tokens: {avg_baseline_tokens:.2f}")
    print(f"Average suffix completion tokens: {avg_suffix_tokens:.2f}")
    print(f"Average suffix length change (tokens): {avg_suffix_length_change:+.2f}")
    if overall_multiplier is not None:
        print(f"Overall token multiplier (suffix/baseline averages): {overall_multiplier:.2f}x")

if __name__ == "__main__":
    asyncio.run(main())
