"""
Utility functions for CatAttack
"""

import json
import time
import subprocess
import psutil
import os
import signal
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from .config import ModelConfig


class VLLMServerManager:
    """Manages vLLM server instances for local model hosting"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.logger = logging.getLogger("catattack.vllm")
    
    def start_server(self, config: ModelConfig, gpu_devices: str = "0,1,2,3,4,5,6,7") -> bool:
        """Start a vLLM server for the given model configuration"""
        
        if config.provider.lower() != "vllm":
            raise ValueError("This method only works with vLLM provider")
        
        server_key = f"{config.model}_{config.port}"
        
        if server_key in self.processes:
            self.logger.info(f"Server for {config.model} already running on port {config.port}")
            return True
        
        cmd = [
            "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--port", str(config.port),
            "--model", config.model,
            "--trust-remote-code",
            "--served-model-name", config.model.split("/")[-1],
            "--host", "0.0.0.0"
        ]
        
        # Add GPU configuration
        if "," in gpu_devices:
            # Multiple GPUs - use tensor parallelism
            num_gpus = len(gpu_devices.split(","))
            cmd.extend(["--tensor-parallel-size", str(num_gpus)])
        
        # Model-specific optimizations
        model_name_lower = config.model.lower()
        if "phi-4" in model_name_lower:
            cmd.extend([
                "--gpu-memory-utilization", "0.90",
                "--max-model-len", "32000",
                "--disable-log-requests"
            ])
        elif "deepseek" in model_name_lower:
            cmd.extend([
                "--gpu-memory-utilization", "0.85",
                "--max-model-len", "8192"
            ])
        elif "qwen" in model_name_lower or "qwq" in model_name_lower:
            cmd.extend([
                "--gpu-memory-utilization", "0.90",
                "--max-model-len", "16384"
            ])
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices
        
        self.logger.info(f"Starting vLLM server for {config.model} on port {config.port}")
        self.logger.debug(f"Command: {' '.join(cmd)}")
        
        # Create log directory
        log_dir = Path("/tmp/vllm_logs")
        log_dir.mkdir(exist_ok=True)
        
        # Start process
        stdout_log = open(log_dir / f"vllm_{config.port}_stdout.log", "w")
        stderr_log = open(log_dir / f"vllm_{config.port}_stderr.log", "w")
        
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_log,
                stderr=stderr_log,
                preexec_fn=os.setsid
            )
            
            self.processes[server_key] = process
            self.logger.info(f"vLLM server started with PID: {process.pid}")
            
            # Wait for server to be ready
            if self.wait_for_server(config, timeout=300):
                self.logger.info(f"vLLM server ready on port {config.port}")
                return True
            else:
                self.logger.error(f"vLLM server failed to start on port {config.port}")
                self.stop_server(config)
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start vLLM server: {str(e)}")
            stdout_log.close()
            stderr_log.close()
            return False
    
    def wait_for_server(self, config: ModelConfig, timeout: int = 300) -> bool:
        """Wait for vLLM server to be ready"""
        import requests
        
        base_url = config.base_url or f"http://localhost:{config.port}"
        health_url = f"{base_url}/v1/models"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(5)
        
        return False
    
    def stop_server(self, config: ModelConfig) -> bool:
        """Stop vLLM server"""
        server_key = f"{config.model}_{config.port}"
        
        if server_key not in self.processes:
            self.logger.warning(f"No server found for {config.model} on port {config.port}")
            return False
        
        process = self.processes[server_key]
        
        try:
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Force kill if not terminated
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
            
            del self.processes[server_key]
            self.logger.info(f"vLLM server stopped for {config.model}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping vLLM server: {str(e)}")
            return False
    
    def stop_all_servers(self):
        """Stop all running vLLM servers"""
        for config_key in list(self.processes.keys()):
            model_name, port = config_key.rsplit("_", 1)
            config = ModelConfig(provider="vllm", model=model_name, port=int(port))
            self.stop_server(config)
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all managed servers"""
        status = {}
        
        for server_key, process in self.processes.items():
            model_name, port = server_key.rsplit("_", 1)
            
            # Check if process is still running
            is_running = process.poll() is None
            
            # Get resource usage if running
            cpu_percent = 0.0
            memory_mb = 0.0
            if is_running:
                try:
                    psutil_process = psutil.Process(process.pid)
                    cpu_percent = psutil_process.cpu_percent()
                    memory_mb = psutil_process.memory_info().rss / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    is_running = False
            
            status[server_key] = {
                "model": model_name,
                "port": int(port),
                "pid": process.pid,
                "running": is_running,
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb
            }
        
        return status


class MetricsCalculator:
    """Calculate various metrics for CatAttack evaluation"""
    
    @staticmethod
    def calculate_attack_success_rate(results: List[Dict[str, Any]]) -> float:
        """Calculate attack success rate"""
        if not results:
            return 0.0
        
        successful = sum(1 for r in results if r.get("attack_successful", False))
        return successful / len(results)
    
    @staticmethod
    def calculate_response_length_increase(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate response length increase metrics"""
        length_increases = []
        
        for result in results:
            original_len = len(result.get("original_question", "").split())
            target_len = len(result.get("target_model_response", "").split())
            if original_len > 0:
                length_increases.append((target_len - original_len) / original_len)
        
        if not length_increases:
            return {"avg_increase": 0.0, "max_increase": 0.0, "pct_longer_1_5x": 0.0}
        
        avg_increase = sum(length_increases) / len(length_increases)
        max_increase = max(length_increases)
        pct_longer_1_5x = sum(1 for inc in length_increases if inc >= 0.5) / len(length_increases)
        
        return {
            "avg_increase": avg_increase,
            "max_increase": max_increase,
            "pct_longer_1_5x": pct_longer_1_5x
        }
    
    @staticmethod
    def calculate_latency_slowdown(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate latency slowdown metrics"""
        latencies = [r.get("total_latency", 0.0) for r in results if r.get("total_latency", 0.0) > 0]
        
        if not latencies:
            return {"avg_latency": 0.0, "max_latency": 0.0, "slowdown_factor": 1.0}
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Estimate baseline latency (this would need to be measured separately)
        baseline_latency = 2.0  # Assume 2 seconds baseline
        slowdown_factor = avg_latency / baseline_latency
        
        return {
            "avg_latency": avg_latency,
            "max_latency": max_latency,
            "slowdown_factor": slowdown_factor
        }
    
    @staticmethod
    def calculate_cost_efficiency(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate cost efficiency metrics"""
        total_cost = sum(r.get("total_cost", 0.0) for r in results)
        successful_attacks = sum(1 for r in results if r.get("attack_successful", False))
        
        if successful_attacks == 0:
            return {"total_cost": total_cost, "cost_per_success": float('inf'), "success_rate": 0.0}
        
        cost_per_success = total_cost / successful_attacks
        success_rate = successful_attacks / len(results) if results else 0.0
        
        return {
            "total_cost": total_cost,
            "cost_per_success": cost_per_success,
            "success_rate": success_rate
        }


class TriggerExtractor:
    """Extract and analyze adversarial triggers"""
    
    @staticmethod
    def extract_trigger(original: str, adversarial: str) -> Dict[str, str]:
        """Extract trigger and determine its type"""
        original = original.strip()
        adversarial = adversarial.strip()
        
        if adversarial == original:
            return {"trigger": "", "type": "none", "position": "none"}
        
        # Check for prefix trigger
        if adversarial.endswith(original):
            prefix = adversarial[:-len(original)].strip()
            if prefix:
                return {"trigger": prefix, "type": "prefix", "position": "start"}
        
        # Check for suffix trigger
        if adversarial.startswith(original):
            suffix = adversarial[len(original):].strip()
            if suffix:
                return {"trigger": suffix, "type": "suffix", "position": "end"}
        
        # Check for infix trigger (more complex)
        # This is a simplified approach - in practice, you might need more sophisticated matching
        return {"trigger": adversarial, "type": "infix", "position": "middle"}
    
    @staticmethod
    def categorize_trigger(trigger: str) -> str:
        """Categorize trigger based on content"""
        trigger_lower = trigger.lower()
        
        # Predefined categories based on the paper
        if any(word in trigger_lower for word in ["cat", "sleep", "fact", "interesting"]):
            return "unrelated_trivia"
        elif any(word in trigger_lower for word in ["save", "invest", "remember", "always"]):
            return "general_advice"
        elif any(word in trigger_lower for word in ["could", "possibly", "around", "maybe"]):
            return "misleading_question"
        elif any(word in trigger_lower for word in ["choose", "select", "option", "answer"]):
            return "multiple_choice"
        else:
            return "other"
    
    @staticmethod
    def analyze_triggers(triggers: List[str]) -> Dict[str, Any]:
        """Analyze a collection of triggers"""
        if not triggers:
            return {"total": 0, "categories": {}, "avg_length": 0.0, "unique_triggers": 0}
        
        categories = {}
        lengths = []
        
        for trigger in triggers:
            category = TriggerExtractor.categorize_trigger(trigger)
            categories[category] = categories.get(category, 0) + 1
            lengths.append(len(trigger.split()))
        
        return {
            "total": len(triggers),
            "categories": categories,
            "avg_length": sum(lengths) / len(lengths),
            "unique_triggers": len(set(triggers))
        }


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("catattack")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    if log_file:
        file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


def save_results_to_hub(results: Dict[str, Any], dataset_name: str, private: bool = True) -> bool:
    """Save results to HuggingFace Hub"""
    try:
        from datasets import Dataset
        
        # Convert results to dataset format
        dataset_data = []
        for result in results.get("attack_results", []):
            dataset_data.append({
                "original_question": result["original_question"],
                "adversarial_question": result["adversarial_question"],
                "ground_truth": result["ground_truth"],
                "proxy_response": result.get("proxy_response"),
                "proxy_correct": result.get("proxy_correct"),
                "target_model_response": result.get("target_model_response"),
                "target_correct": result.get("target_correct"),
                "attack_successful": result["attack_successful"],
                "iterations": result["iterations"],
                "trigger": result.get("extracted_trigger", ""),
                "total_cost": result["total_cost"],
                "total_latency": result["total_latency"]
            })
        
        if not dataset_data:
            return False
        
        # Create dataset and push to hub
        dataset = Dataset.from_list(dataset_data)
        dataset.push_to_hub(dataset_name, private=private)
        
        return True
        
    except Exception as e:
        logging.getLogger("catattack").error(f"Failed to push to hub: {str(e)}")
        return False


def estimate_gpu_memory_requirements(model_name: str) -> Dict[str, Any]:
    """Estimate GPU memory requirements for a model"""
    
    # Rough estimates based on model size (in GB)
    model_sizes = {
        "7b": 14,   # 7B model needs ~14GB
        "8b": 16,   # 8B model needs ~16GB  
        "13b": 26,  # 13B model needs ~26GB
        "30b": 60,  # 30B model needs ~60GB
        "32b": 64,  # 32B model needs ~64GB
        "70b": 140, # 70B model needs ~140GB
    }
    
    model_lower = model_name.lower()
    
    # Extract model size from name
    estimated_size = 16  # Default
    for size_key, memory_gb in model_sizes.items():
        if size_key in model_lower:
            estimated_size = memory_gb
            break
    
    # Calculate recommended GPU configuration
    gpu_memory_per_card = 80  # A100 80GB
    num_gpus_needed = max(1, (estimated_size + gpu_memory_per_card - 1) // gpu_memory_per_card)
    
    return {
        "estimated_memory_gb": estimated_size,
        "recommended_gpus": num_gpus_needed,
        "gpu_memory_per_card": gpu_memory_per_card,
        "tensor_parallel_size": num_gpus_needed if num_gpus_needed <= 8 else 8
    }
