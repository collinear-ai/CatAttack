#!/usr/bin/env python3
"""
CatAttack Suffix Generation Pipeline
Simplified version focused only on generating adversarial suffixes
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import load_config
from src.catattack import CatAttack, CatAttackResults
from src.dataset import create_sample_dataset

from dotenv import load_dotenv

load_dotenv()  # reads .env and populates os.environ


def main():
    """Run the suffix generation pipeline"""
    print("🐱 CatAttack: Suffix Generation Pipeline")
    print("=" * 50)

    config = load_config("config.yaml")

    problems = create_sample_dataset(config.dataset.num_problems)

    print(f"Loaded {len(problems)} sample problems")
    print(f"Using proxy target model: {config.models['proxy_target'].model}")
    print(f"Evaluating on target model: {config.models['target_model'].model}")
    print(f"Max iterations per problem: {config.attack.max_iterations}")
    print()

    catattack = CatAttack(config)

    run_start_time = time.time()

    results_summary = catattack.run_attack(problems)

    successful_suffixes = [r.extracted_trigger for r in results_summary.attack_results if r.attack_successful]

    saved_path = catattack.save_results(results_summary)
    print(f"📁 Saved detailed results to {saved_path}")

    if config.output.push_to_hub:
        hub_dataset = catattack.save_modified_problems_to_hub(results_summary)
        if hub_dataset:
            print(f"🌐 Uploaded dataset to https://huggingface.co/datasets/{hub_dataset}")

    print(f"\n🎯 RESULTS SUMMARY")
    print(f"Successful attacks: {len(successful_suffixes)}/{len(problems)}")
    print(f"Success rate: {len(successful_suffixes)/len(problems):.1%}")

    if successful_suffixes:
        print(f"\n🔥 DISCOVERED SUFFIXES:")
        for i, suffix in enumerate(successful_suffixes, 1):
            print(f"{i}. {suffix}")

    return successful_suffixes


if __name__ == "__main__":
    main()
