"""CatAttack suffix generation CLI."""

import time

from dotenv import load_dotenv

from .. import CatAttack
from ..config import load_config
from ..dataset import load_dataset, create_sample_dataset


def main(config_path: str = "config.yaml"):
    """Run the suffix generation pipeline."""
    load_dotenv()

    print("🐱 CatAttack: Suffix Generation Pipeline")
    print("=" * 50)

    config = load_config(config_path)
    
    # Load dataset from config, or use hardcoded samples as fallback
    if config.dataset.name or config.dataset.local_path:
        problems = load_dataset(config.dataset)
        print(f"Loaded {len(problems)} problems from {config.dataset.name or config.dataset.local_path}")
    else:
        problems = create_sample_dataset(config.dataset.num_problems or 10)
        print(f"Loaded {len(problems)} sample problems (hardcoded fallback)")
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
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_file)
