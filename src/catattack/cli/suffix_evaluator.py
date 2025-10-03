"""Suffix evaluator for CatAttack."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from ..config import load_config, CatAttackConfig
from .. import CatAttack
from ..dataset import load_dataset
from ..manual_suffixes import MANUAL_SUFFIXES

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
    load_dotenv()

    catattack = CatAttack(config)

    test_dataset_config = config.test_dataset
    problems = load_dataset(test_dataset_config)

    if config.evaluation.num_problems:
        problems = problems[: config.evaluation.num_problems]

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
            "overall_token_multiplier": overall_multiplier,
        },
        "details": [
            {
                "problem_index": r.problem_index,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "baseline_responses": r.baseline_responses,
                "baseline_correctness": r.baseline_correctness,
                "baseline_completion_tokens": r.baseline_completion_tokens,
                "suffix_questions": r.suffix_questions,
                "suffix_correctness": r.suffix_correctness,
                "suffix_responses": r.suffix_responses,
                "suffix_completion_tokens": r.suffix_completion_tokens,
                "suffix_token_multipliers": r.suffix_token_multipliers,
            }
            for r in evaluation_results
        ],
    }


def main(config_path: str = "config.yaml", output_path: Optional[str] = None):
    load_dotenv()

    config = load_config(config_path)
    results = asyncio.run(evaluate_suffixes(config))

    results_dir = Path(config.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = results_dir / config.evaluation.results_file
    else:
        output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary metrics
    summary = results["summary"]
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n📋 Test Configuration:")
    print(f"   Total problems: {summary['total_problems']}")
    print(f"   Runs per suffix: {summary['num_runs']}")
    print(f"   Total evaluations: {summary['total_evaluations']}")
    
    print(f"\n✅ Baseline Performance (without suffixes):")
    print(f"   Accuracy: {summary['baseline_accuracy']:.1%}")
    print(f"   Error Rate: {summary['baseline_error_rate']:.1%}")
    print(f"   Avg Completion Tokens: {summary['avg_baseline_completion_tokens']:.1f}")
    
    print(f"\n🎯 Trigger-wise Performance:")
    for idx, suffix in enumerate(MANUAL_SUFFIXES):
        acc = summary['suffix_accuracy_per_trigger'][idx]
        err = summary['suffix_error_rate_per_trigger'][idx]
        print(f"   Trigger {idx+1}: \"{suffix}\"")
        print(f"      Accuracy: {acc:.1%} | Error Rate: {err:.1%}")
    
    print(f"\n📈 Combined Suffix Performance:")
    print(f"   Combined Accuracy: {summary['combined_suffix_accuracy']:.1%}")
    print(f"   Combined Error Rate: {summary['combined_suffix_error_rate']:.1%}")
    if summary['catattack_asr'] is not None:
        print(f"   CatAttack ASR: {summary['catattack_asr']:.1%}")
    else:
        print(f"   CatAttack ASR: N/A (baseline had no errors)")
    print(f"   Avg Completion Tokens: {summary['avg_suffix_completion_tokens']:.1f}")
    
    print(f"\n🔢 Token Length Analysis:")
    print(f"   Baseline avg tokens: {summary['avg_baseline_completion_tokens']:.1f}")
    print(f"   Suffix avg tokens: {summary['avg_suffix_completion_tokens']:.1f}")
    print(f"   Change: {summary['avg_suffix_length_change']:+.1f} tokens")
    if summary['overall_token_multiplier']:
        print(f"   Multiplier: {summary['overall_token_multiplier']:.2f}x")
    
    print("\n" + "=" * 60)
    print(f"💾 Saved evaluation results to {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_file)
