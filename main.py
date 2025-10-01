#!/usr/bin/env python3
"""
CatAttack: Query-Agnostic Adversarial Triggers for Reasoning Models

Main entry point for running CatAttack experiments.
Based on the paper: "Cats Confuse Reasoning LLM: Query-Agnostic Adversarial Triggers for Reasoning Models"
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import load_config
from src.catattack import CatAttack
from src.dataset import create_sample_dataset, DATASET_CONFIGS
from src.utils import setup_logging, VLLMServerManager

from dotenv import load_dotenv

load_dotenv()  # reads .env and populates os.environ


async def main():
    parser = argparse.ArgumentParser(description="CatAttack: Adversarial Triggers for Reasoning Models")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--dataset", "-d", help="Dataset name (overrides config)")
    parser.add_argument("--num-problems", "-n", type=int, help="Number of problems to process")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--sample", action="store_true", help="Use sample dataset for testing")
    parser.add_argument("--start-servers", action="store_true", help="Start vLLM servers automatically")
    parser.add_argument("--stop-servers", action="store_true", help="Stop vLLM servers after completion")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without running attack")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Setup logging
    logger = setup_logging(config.logging.level, config.logging.log_file)
    logger.info("Starting CatAttack")
    
    # Override dataset if specified
    if args.dataset:
        if args.dataset in DATASET_CONFIGS:
            config.dataset = DATASET_CONFIGS[args.dataset]
        else:
            config.dataset.name = args.dataset
    
    # Override number of problems
    if args.num_problems:
        config.dataset.num_problems = args.num_problems
    
    # Dry run - just validate configuration
    if args.dry_run:
        logger.info("Dry run mode - validating configuration")
        try:
            config.validate()
            logger.info("Configuration is valid")
            
            # Test model connections
            catattack = CatAttack(config)
            logger.info("Model clients initialized successfully")
            
            return 0
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return 1
    
    # Server management
    server_manager = VLLMServerManager() if args.start_servers else None
    
    try:
        # Start vLLM servers if requested
        if args.start_servers:
            logger.info("Starting vLLM servers...")
            
            for model_type, model_config in config.models.items():
                if model_config.provider.lower() == "vllm":
                    logger.info(f"Starting server for {model_type}: {model_config.model}")
                    success = server_manager.start_server(model_config)
                    if not success:
                        logger.error(f"Failed to start server for {model_config.model}")
                        return 1
        
        # Prepare dataset
        if args.sample:
            logger.info("Using sample dataset")
            problems = create_sample_dataset(config.dataset.num_problems)
        else:
            problems = None  # Will be loaded by CatAttack
        
        # Initialize CatAttack
        catattack = CatAttack(config)
        
        # Run the attack
        logger.info("Running CatAttack...")
        results = await catattack.run_attack(problems)
        
        # Save results
        output_path = args.output
        if not output_path:
            output_path = catattack.save_results(results)
        else:
            output_path = catattack.save_results(results, output_path)
        
        # Print summary
        print("\n" + "="*60)
        print("CatAttack Results Summary")
        print("="*60)
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
        
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error running CatAttack: {e}", exc_info=True)
        return 1
    
    finally:
        # Stop servers if we started them
        if server_manager and args.stop_servers:
            logger.info("Stopping vLLM servers...")
            server_manager.stop_all_servers()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
