"""
Core CatAttack implementation
Based on the paper: "Cats Confuse Reasoning LLM: Query-Agnostic Adversarial Triggers for Reasoning Models"
"""

import asyncio
import json
import logging
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

def _progress(iterable, description: str = "", total: Optional[int] = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=description, total=total)

from .config import CatAttackConfig
from .models import ModelManager, ModelResponse
from .dataset import load_dataset
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from prompts.attacker import ATTACKER_SYSTEM_PROMPT, ATTACKER_PROMPT_TEMPLATE
from prompts.judge import JUDGE_SYSTEM_PROMPT, JUDGE_PROMPT_TEMPLATE


@dataclass
class AttackResult:
    """Result of a single attack attempt"""
    original_question: str
    adversarial_question: str
    ground_truth: str
    proxy_response: str
    target_response: Optional[str] = None
    attack_successful: bool = False
    iterations: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    trigger_type: str = "suffix"
    extracted_trigger: str = ""
    attempts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CatAttackResults:
    """Results from a CatAttack run"""
    attack_results: List[AttackResult]
    total_cost: float
    total_time: float
    attack_success_rate: float
    avg_iterations: float
    successful_triggers: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "attack_results": [asdict(result) for result in self.attack_results],
            "total_cost": self.total_cost,
            "total_time": self.total_time,
            "attack_success_rate": self.attack_success_rate,
            "avg_iterations": self.avg_iterations,
            "successful_triggers": self.successful_triggers
        }


class CatAttack:
    """Main CatAttack class implementing the iterative attack pipeline"""

    def __init__(self, config: CatAttackConfig):
        self.config = config
        self.model_manager = ModelManager()
        self.logger = self._setup_logging()

        self.attacker_client = self.model_manager.get_client(config.get_model_config("attacker"))
        self.proxy_client = self.model_manager.get_client(config.get_model_config("proxy_target"))
        self.judge_client = self.model_manager.get_client(config.get_model_config("judge"))

        if "target" in config.models:
            self.target_client = self.model_manager.get_client(config.get_model_config("target"))
        else:
            self.target_client = None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("catattack")
        logger.setLevel(getattr(logging, self.config.logging.level))
        
        # File handler
        fh = logging.FileHandler(self.config.logging.log_file)
        fh.setLevel(getattr(logging, self.config.logging.level))
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    def run_attack(self, problems: Optional[List[Dict]] = None) -> CatAttackResults:
        """Run the complete CatAttack pipeline using threads"""
        start_time = time.time()

        if problems is None:
            problems = load_dataset(self.config.dataset)

        total_problems = len(problems)
        self.logger.info(f"Starting CatAttack on {total_problems} problems")

        results_list: List[Optional[AttackResult]] = [None] * total_problems
        successful_triggers: List[str] = []

        num_threads = max(1, getattr(self.config.attack, "num_threads", 1))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self._attack_single_problem_sync, problem) for problem in problems]
            iterator = _progress(range(total_problems), description="Attacking problems", total=total_problems)

            for idx in iterator:
                future = futures[idx]
                try:
                    result = future.result()
                    if result:
                        results_list[idx] = result
                        if result.attack_successful and result.extracted_trigger:
                            successful_triggers.append(result.extracted_trigger)
                except Exception as e:
                    self.logger.error(f"Error processing problem {idx + 1}: {str(e)}")

        attack_results = [r for r in results_list if r is not None]

        total_time = time.time() - start_time

        successful_attacks = [r for r in attack_results if r.attack_successful]
        attack_success_rate = len(successful_attacks) / len(attack_results) if attack_results else 0.0
        avg_iterations = sum(r.iterations for r in attack_results) / len(attack_results) if attack_results else 0.0

        results = CatAttackResults(
            attack_results=attack_results,
            total_cost=self.model_manager.get_total_cost(),
            total_time=total_time,
            attack_success_rate=attack_success_rate,
            avg_iterations=avg_iterations,
            successful_triggers=list(set(successful_triggers))
        )

        self.logger.info(f"CatAttack completed: {attack_success_rate:.2%} success rate, "
                        f"${results.total_cost:.2f} cost, {results.total_time:.1f}s")

        if self.config.output.push_to_hub:
            hub_dataset = self.save_modified_problems_to_hub(results)
            if hub_dataset:
                self.logger.info(f"Modified problems available at: https://huggingface.co/datasets/{hub_dataset}")

        return results

    def _attack_single_problem_sync(self, problem: Dict[str, str]) -> AttackResult:
        return asyncio.run(self.attack_single_problem(problem))

    async def attack_single_problem(self, problem: Dict[str, str]) -> AttackResult:
        """Attack a single math problem"""
        original_question = problem[self.config.dataset.problem_field]
        ground_truth = problem[self.config.dataset.answer_field]
        
        self.logger.debug(f"Attacking problem: {original_question[:100]}...")
        
        # Initialize result
        result = AttackResult(
            original_question=original_question,
            adversarial_question=original_question,
            ground_truth=ground_truth,
            proxy_response="",
        )
        
        # Test baseline (original question)
        baseline_response = await self.proxy_client.generate(original_question)
        baseline_correct = await self.judge_answer(original_question, ground_truth, baseline_response.content)
        
        if not baseline_correct:
            # Model already gets it wrong, no attack needed
            result.attack_successful = True
            result.proxy_response = baseline_response.content
            return result
        
        # Iterative attack
        current_question = original_question
        attack_history = []

        for iteration in range(self.config.attack.max_iterations):
            self.logger.debug(f"Attack iteration {iteration + 1}")
            
            # Generate adversarial question
            adversarial_question = await self.generate_adversarial_question(
                current_question, ground_truth
            )
            
            if not adversarial_question or adversarial_question == current_question:
                self.logger.debug("No new adversarial question generated")
                break
            
            # Test on proxy target
            proxy_response = await self.proxy_client.generate(adversarial_question)
            is_correct = await self.judge_answer(adversarial_question, ground_truth, proxy_response.content)
            
            # Update result
            result.iterations = iteration + 1
            result.adversarial_question = adversarial_question
            result.proxy_response = proxy_response.content
            result.total_cost += proxy_response.cost
            result.total_latency += proxy_response.latency
            
            attempt_record = {
                "iteration": iteration + 1,
                "question": adversarial_question,
                "response": proxy_response.content,
                "correct": is_correct,
                "judge_feedback": "Correct" if is_correct else "Incorrect",
                "total_cost": result.total_cost,
                "total_latency": result.total_latency,
            }
            result.attempts.append(attempt_record)
            
            if not is_correct:
                # Attack successful!
                result.attack_successful = True
                result.extracted_trigger = self.extract_trigger(original_question, adversarial_question)
                self.logger.debug(f"Attack successful after {iteration + 1} iterations")
                break
            
            current_question = adversarial_question
        
        # Test on target model if available and attack was successful
        if result.attack_successful and self.target_client:
            target_response = await self.target_client.generate(result.adversarial_question)
            result.target_response = target_response.content
            result.total_cost += target_response.cost
            result.total_latency += target_response.latency
        
        return result
    
    async def generate_adversarial_question(
        self, 
        question: str, 
        ground_truth: str
    ) -> str:
        """Generate adversarial version of the question"""
        
        try:
            from jinja2 import Template
        except ImportError:
            self.logger.error("jinja2 not installed. Install with: pip install jinja2")
            return ""
        
        # Prepare revision history in the format expected by Jinja2 template
        revision_history = []
        # The attack_history list is removed, so we can't pass it here.
        # This part of the code will need to be refactored if attack_history is truly removed.
        # For now, we'll just pass an empty list or a placeholder.
        # Assuming attack_history is no longer needed for the prompt template.
        
        # Use Jinja2 to render the prompt template
        template = Template(ATTACKER_PROMPT_TEMPLATE)
        prompt = template.render(
            original_question=question,
            ground_truth_answer=ground_truth,
            revision_history=[] # Pass an empty list as attack_history is removed
        )
        
        try:
            response = await self.attacker_client.generate(prompt, temperature=0.7)
            
            # Try to parse JSON response
            response.content = response.content.replace("```json", "").replace("```", "")
            try:
                import json
                result = json.loads(response.content)
                return result.get("final_question", result.get("question", ""))
            except json.JSONDecodeError:
                # Fallback: return the raw response
                self.logger.warning("Could not parse JSON from attacker response")
                return response.content.strip()
                
        except Exception as e:
            self.logger.error(f"Error generating adversarial question: {str(e)}")
            return ""
    
    async def judge_answer(self, question: str, ground_truth: str, student_answer: str) -> bool:
        """Judge if the student answer is correct"""
        try:
            from jinja2 import Template
        except ImportError:
            self.logger.error("jinja2 not installed. Install with: pip install jinja2")
            return True  # Default to correct on error
        
        # Use Jinja2 to render the prompt template
        template = Template(JUDGE_PROMPT_TEMPLATE)
        prompt = template.render(
            question=question,
            ground_truth_answer=ground_truth,
            student_answer=student_answer
        )
        
        try:
            response = await self.judge_client.generate(prompt, temperature=0.0)
            response = response.content
            response = response.replace("```json", "").replace("```", "")
            
            # Try to parse JSON response
            try:
                result = json.loads(response)
                # The judge returns 1 for correct, 0 for incorrect
                output = result.get("output", 1)
                if isinstance(output, str):
                    output = int(output.strip().split()[0])  # Extract number from string
                return output == 1
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Could not parse judge response as JSON: {e}")
                # Fallback: look for output indicators in text
                content_lower = response.lower()
                if '"output": 0' in content_lower or '"output":0' in content_lower:
                    return False
                elif '"output": 1' in content_lower or '"output":1' in content_lower:
                    return True
                else:
                    # Default to correct if unclear
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error judging answer: {str(e)}")
            return True  # Default to correct on error
    
    def extract_trigger(self, original: str, adversarial: str) -> str:
        """Extract the suffix trigger from adversarial question"""
        # CatAttack only uses suffixes
        if adversarial.startswith(original):
            # Suffix trigger
            return adversarial[len(original):].strip()
        else:
            # If it doesn't start with original, something went wrong
            return ""
    
    async def evaluate_triggers(self, triggers: List[str], test_problems: List[Dict]) -> Dict[str, float]:
        """Evaluate extracted triggers on test problems"""
        self.logger.info(f"Evaluating {len(triggers)} triggers on {len(test_problems)} problems")
        
        results = {}
        
        for trigger in triggers:
            successful_attacks = 0
            
            for problem in test_problems:
                original_question = problem[self.config.dataset.problem_field]
                ground_truth = problem[self.config.dataset.answer_field]
                
                # Apply suffix trigger
                adversarial_question = f"{original_question} {trigger}"
                
                # Test on target model
                if self.target_client:
                    response = await self.target_client.generate(adversarial_question)
                    is_correct = await self.judge_answer(adversarial_question, ground_truth, response.content)
                    
                    if not is_correct:
                        successful_attacks += 1
            
            success_rate = successful_attacks / len(test_problems) if test_problems else 0.0
            results[trigger] = success_rate
        
        return results
    
    def save_results(self, results: CatAttackResults, filename: Optional[str] = None) -> str:
        """Save results to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"catattack_results_{timestamp}.json"
        
        results_path = Path(self.config.output.results_dir) / filename
        
        with open(results_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
        return str(results_path)

    def save_modified_problems_to_hub(self, results: CatAttackResults) -> Optional[str]:
        """
        Save all successfully modified problems (where judge detected incorrect answer) 
        to HuggingFace dataset
        """
        if not self.config.output.push_to_hub or not self.config.output.hub_dataset_name:
            self.logger.info("Skipping HuggingFace upload (push_to_hub=False or no dataset name)")
            return None
        
        try:
            from datasets import Dataset
        except ImportError:
            self.logger.error("datasets library not installed. Install with: pip install datasets")
            return None
        
        # Filter for successful attacks (or include all if configured)
        modified_problems = []
        include_failed = self.config.output.include_failed_attacks

        for result in results.attack_results:
            if result.attack_successful or include_failed:
                modified_problems.append({
                    "original_question": result.original_question,
                    "modified_question": result.adversarial_question,
                    "ground_truth": result.ground_truth,
                    "extracted_trigger": result.extracted_trigger,
                    "proxy_model_response": result.proxy_response,
                    "target_model_response": result.target_response if result.target_response else "",
                    "iterations": result.iterations,
                    "total_cost": result.total_cost,
                    "total_latency": result.total_latency,
                    "attack_successful": result.attack_successful
                })
        
        if not modified_problems:
            self.logger.warning("No successful attacks to upload")
            return None
        
        self.logger.info(f"Uploading {len(modified_problems)} modified problems to HuggingFace Hub")
        
        # Create dataset
        dataset = Dataset.from_list(modified_problems)
        
        # Push to hub
        try:
            dataset.push_to_hub(
                self.config.output.hub_dataset_name,
                private=self.config.output.hub_private
            )
            self.logger.info(f"Successfully uploaded to {self.config.output.hub_dataset_name}")
            return self.config.output.hub_dataset_name
        except Exception as e:
            self.logger.error(f"Failed to push to HuggingFace Hub: {str(e)}")
            return None
