"""
Model interfaces and clients for CatAttack
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import openai
import anthropic
import requests
from dataclasses import dataclass

from .config import ModelConfig


@dataclass
class ModelResponse:
    """Response from a model"""
    content: str
    tokens_used: int = 0
    latency: float = 0.0
    cost: float = 0.0
    raw_response: Optional[Dict] = None


class BaseModelClient(ABC):
    """Abstract base class for model clients"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.total_cost = 0.0
        self.total_tokens = 0
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from the model"""
        pass
    
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on tokens (implement per provider)"""
        return 0.0


class OpenAIClient(BaseModelClient):
    """OpenAI API client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(
            api_key=config.get_api_key(),
            base_url=config.base_url
        )
        
        # Pricing per 1K tokens (approximate, update as needed)
        self.pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using OpenAI API"""
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]}
            )
            
            latency = time.time() - start_time
            content = response.choices[0].message.content
            
            # Calculate tokens and cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            cost = self.estimate_cost_openai(input_tokens, output_tokens)
            self.total_cost += cost
            self.total_tokens += total_tokens
            
            return ModelResponse(
                content=content,
                tokens_used=total_tokens,
                latency=latency,
                cost=cost,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def estimate_cost_openai(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for OpenAI models"""
        model_name = self.config.model.lower()
        if model_name in self.pricing:
            prices = self.pricing[model_name]
            return (input_tokens / 1000 * prices["input"] + 
                   output_tokens / 1000 * prices["output"])
        return 0.0


class AnthropicClient(BaseModelClient):
    """Anthropic API client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(
            api_key=config.get_api_key()
        )
        
        # Pricing per 1K tokens
        self.pricing = {
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
        }
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using Anthropic API"""
        start_time = time.time()
        
        try:
            response = await self.client.messages.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature)
            )
            
            latency = time.time() - start_time
            content = response.content[0].text
            
            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            
            cost = self.estimate_cost_anthropic(input_tokens, output_tokens)
            self.total_cost += cost
            self.total_tokens += total_tokens
            
            return ModelResponse(
                content=content,
                tokens_used=total_tokens,
                latency=latency,
                cost=cost,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")
    
    def estimate_cost_anthropic(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for Anthropic models"""
        model_name = self.config.model.lower()
        if model_name in self.pricing:
            prices = self.pricing[model_name]
            return (input_tokens / 1000 * prices["input"] + 
                   output_tokens / 1000 * prices["output"])
        return 0.0


class VLLMClient(BaseModelClient):
    """vLLM server client (OpenAI-compatible API)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.base_url or f"http://localhost:{config.port}/v1"
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using vLLM server"""
        start_time = time.time()
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False
        }
        
        try:
            # Prepare headers with API key if available
            headers = {"Content-Type": "application/json"}
            api_key = self.config.get_api_key()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
                
            data = response.json()
            latency = time.time() - start_time
            
            content = data["choices"][0]["message"]["content"]
            
            # vLLM might not return usage info, estimate tokens
            tokens_used = len(prompt.split()) + len(content.split())  # Rough estimate
            
            return ModelResponse(
                content=content,
                tokens_used=tokens_used,
                latency=latency,
                cost=0.0,  # Local inference, no cost
                raw_response=data
            )
            
        except Exception as e:
            raise RuntimeError(f"vLLM server error: {str(e)}")


class SGLangClient(BaseModelClient):
    """SGLang server client (OpenAI-compatible API)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.base_url or f"http://localhost:{config.port}/v1"
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using SGLang server"""
        start_time = time.time()
        
        payload = {
            "model": self.config.model.split("/")[-1],  # SGLang uses model name, not full path
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False
        }
        
        try:
            # Prepare headers with API key if available
            headers = {"Content-Type": "application/json"}
            api_key = self.config.get_api_key()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
                
            data = response.json()
            latency = time.time() - start_time
            
            content = data["choices"][0]["message"]["content"]
            
            # SGLang might not return usage info, estimate tokens
            tokens_used = len(prompt.split()) + len(content.split())  # Rough estimate
            
            return ModelResponse(
                content=content,
                tokens_used=tokens_used,
                latency=latency,
                cost=0.0,  # Local inference, no cost
                raw_response=data
            )
            
        except Exception as e:
            raise RuntimeError(f"SGLang server error: {str(e)}")


class BedrockClient(BaseModelClient):
    """AWS Bedrock client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import boto3
            self.client = boto3.client('bedrock-runtime')
        except ImportError:
            raise ImportError("boto3 is required for Bedrock client")
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using AWS Bedrock"""
        start_time = time.time()
        
        # This is a simplified implementation - actual Bedrock integration
        # would need proper model-specific payload formatting
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature)
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.config.model,
                body=json.dumps(payload)
            )
            
            latency = time.time() - start_time
            response_body = json.loads(response['body'].read())
            
            # Extract content (format depends on specific Bedrock model)
            content = response_body.get('completion', '')
            
            return ModelResponse(
                content=content,
                tokens_used=0,  # Bedrock doesn't always return token counts
                latency=latency,
                cost=0.0,  # Would need to implement Bedrock pricing
                raw_response=response_body
            )
            
        except Exception as e:
            raise RuntimeError(f"Bedrock API error: {str(e)}")


class ModelManager:
    """Manages model clients"""
    
    def __init__(self):
        self.clients: Dict[str, BaseModelClient] = {}
    
    def get_client(self, config: ModelConfig) -> BaseModelClient:
        """Get or create a client for the given configuration"""
        client_key = f"{config.provider}_{config.model}"
        
        if client_key not in self.clients:
            self.clients[client_key] = self._create_client(config)
        
        return self.clients[client_key]
    
    def _create_client(self, config: ModelConfig) -> BaseModelClient:
        """Create a new client based on provider"""
        provider = config.provider.lower()
        
        if provider == "openai":
            return OpenAIClient(config)
        elif provider == "anthropic":
            return AnthropicClient(config)
        elif provider == "vllm":
            return VLLMClient(config)
        elif provider == "sglang":
            return SGLangClient(config)
        elif provider == "bedrock":
            return BedrockClient(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def get_total_cost(self) -> float:
        """Get total cost across all clients"""
        return sum(client.total_cost for client in self.clients.values())
    
    def get_total_tokens(self) -> int:
        """Get total tokens across all clients"""
        return sum(client.total_tokens for client in self.clients.values())
