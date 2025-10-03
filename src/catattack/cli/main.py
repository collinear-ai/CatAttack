"""Main CLI for CatAttack."""

import argparse
import asyncio

from dotenv import load_dotenv

from .. import CatAttack
from ..config import load_config
from ..dataset import create_sample_dataset, DATASET_CONFIGS
from ..utils import setup_logging, VLLMServerManager


async def run_main(args=None):
    """Run CatAttack via CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="CatAttack: Adversarial Triggers for Reasoning Models")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--dataset", "-d", help="Dataset name (overrides config)")
    parser.add_argument("--num-problems", "-n", type=int, help="Number of problems to process")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--sample", action="store_true", help="Use sample dataset for testing")
    parser.add_argument("--start-servers", action="store_true", help="Start vLLM servers automatically")
    parser.add_argument("--stop-servers", action="store_true", help="Stop vLLM servers after completion")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without running attack")

    parsed_args = parser.parse_args(args=args)

    try:
        config = load_config(parsed_args.config)
    except Exception as e:  # pragma: no cover - CLI error path
        print(f"Error loading configuration: {e}")
        return 1

    logger = setup_logging(config.logging.level, config.logging.log_file)
    logger.info("Starting CatAttack")

    if parsed_args.dataset:
        if parsed_args.dataset in DATASET_CONFIGS:
            config.dataset = DATASET_CONFIGS[parsed_args.dataset]
        else:
            config.dataset.name = parsed_args.dataset

    if parsed_args.num_problems:
        config.dataset.num_problems = parsed_args.num_problems

    if parsed_args.dry_run:
        logger.info("Dry run mode - validating configuration")
        try:
            config.validate()
            logger.info("Configuration is valid")

            catattack = CatAttack(config)
            logger.info("Model clients initialized successfully")

            return 0
        except Exception as e:  # pragma: no cover - CLI error path
            logger.error(f"Configuration validation failed: {e}")
            return 1

    server_manager = VLLMServerManager() if parsed_args.start_servers else None

    try:
        if parsed_args.start_servers:
            logger.info("Starting vLLM servers...")

            for model_type, model_config in config.models.items():
                if model_config.provider.lower() == "vllm":
                    logger.info(f"Starting server for {model_type}: {model_config.model}")
                    success = server_manager.start_server(model_config)
                    if not success:
                        logger.error(f"Failed to start server for {model_config.model}")
                        return 1

        if parsed_args.sample:
            logger.info("Using sample dataset")
            problems = create_sample_dataset(config.dataset.num_problems)
        else:
            problems = None

        catattack = CatAttack(config)

        logger.info("Running CatAttack...")
        results = catattack.run_attack(problems)

        output_path = parsed_args.output
        if not output_path:
            output_path = catattack.save_results(results)
        else:
            output_path = catattack.save_results(results, output_path)

        print("\n" + "=" * 60)
        print("CatAttack Results Summary")
        print("=" * 60)
        print(f"Total problems processed: {len(results.attack_results)}")
        print(f"Attack success rate: {results.attack_success_rate:.2%}")
        print(f"Average iterations: {results.avg_iterations:.1f}")
        print(f"Total cost: ${results.total_cost:.2f}")
        print(f"Total time: {results.total_time:.1f}s")
        print(f"Successful triggers found: {len(results.successful_triggers)}")
        print(f"Results saved to: {output_path}")

        if results.successful_triggers:
            print("\nTop 5 Successful Triggers:")
            for i, trigger in enumerate(results.successful_triggers[:5], 1):
                print(f"{i}. {trigger}")

        if config.output.push_to_hub and results.attack_success_rate > 0:
            print(f"\n📤 Modified problems uploaded to HuggingFace:")
            print(f"   https://huggingface.co/datasets/{config.output.hub_dataset_name}")

        print("=" * 60)

        return 0

    except KeyboardInterrupt:  # pragma: no cover - CLI path
        logger.info("Interrupted by user")
        return 1
    except Exception as e:  # pragma: no cover - CLI error path
        logger.error(f"Error running CatAttack: {e}", exc_info=True)
        return 1
    finally:
        if server_manager and parsed_args.stop_servers:
            logger.info("Stopping vLLM servers...")
            server_manager.stop_all_servers()


def main(args=None):
    """Synchronous entry point for console scripts."""
    return asyncio.run(run_main(args=args))
