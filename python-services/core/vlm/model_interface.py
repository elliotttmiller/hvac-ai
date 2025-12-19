"""
HVAC VLM Model Interface

Provides a unified interface for HVAC-specific Vision-Language Models,
supporting both Qwen2-VL and InternVL backends.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from PIL import Image

from .data_schema import HVACDataSchema, HVACComponentType

logger = logging.getLogger(__name__)


class HVACVLMInterface(ABC):
    """Abstract interface for HVAC VLM models"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        quantization: Optional[str] = None
    ):
        """
        Initialize VLM interface
        
        Args:
            model_path: Path to the model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            quantization: Quantization mode ('int8', 'int4', or None)
        """
        self.model_path = model_path
        self.device = device
        self.quantization = quantization
        self.model = None
        self.processor = None
        self.schema = HVACDataSchema()
        
    @abstractmethod
    def load_model(self):
        """Load the VLM model"""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: Union[str, Image.Image]) -> Any:
        """Preprocess image for model input"""
        pass
    
    @abstractmethod
    def generate_response(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """Generate model response"""
        pass
    
    def analyze_components(
        self,
        image_path: str,
        return_json: bool = True
    ) -> Union[Dict, str]:
        """
        Analyze HVAC components in a blueprint
        
        Args:
            image_path: Path to the blueprint image
            return_json: Whether to return parsed JSON or raw text
            
        Returns:
            Dictionary of detected components or raw response text
        """
        prompt = self.schema.get_prompt_templates()["component_detection"]
        response = self.generate_response(image_path, prompt)
        
        if return_json:
            try:
                return self._parse_json_response(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {"error": "Failed to parse response", "raw_response": response}
        return response
    
    def analyze_relationships(
        self,
        image_path: str,
        return_json: bool = True
    ) -> Union[Dict, str]:
        """
        Analyze system relationships in a blueprint
        
        Args:
            image_path: Path to the blueprint image
            return_json: Whether to return parsed JSON or raw text
            
        Returns:
            Dictionary of relationships and violations or raw response text
        """
        prompt = self.schema.get_prompt_templates()["relationship_analysis"]
        response = self.generate_response(image_path, prompt)
        
        if return_json:
            try:
                return self._parse_json_response(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {"error": "Failed to parse response", "raw_response": response}
        return response
    
    def extract_specifications(
        self,
        image_path: str,
        return_json: bool = True
    ) -> Union[Dict, str]:
        """
        Extract engineering specifications from blueprint
        
        Args:
            image_path: Path to the blueprint image
            return_json: Whether to return parsed JSON or raw text
            
        Returns:
            Dictionary of specifications or raw response text
        """
        prompt = self.schema.get_prompt_templates()["specification_extraction"]
        response = self.generate_response(image_path, prompt)
        
        if return_json:
            try:
                return self._parse_json_response(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {"error": "Failed to parse response", "raw_response": response}
        return response
    
    def check_code_compliance(
        self,
        image_path: str,
        return_json: bool = True
    ) -> Union[Dict, str]:
        """
        Check HVAC design for code compliance
        
        Args:
            image_path: Path to the blueprint image
            return_json: Whether to return parsed JSON or raw text
            
        Returns:
            Dictionary of compliance issues or raw response text
        """
        prompt = self.schema.get_prompt_templates()["code_compliance"]
        response = self.generate_response(image_path, prompt)
        
        if return_json:
            try:
                return self._parse_json_response(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {"error": "Failed to parse response", "raw_response": response}
        return response
    
    def custom_analysis(
        self,
        image_path: str,
        custom_prompt: str,
        return_json: bool = False
    ) -> Union[Dict, str]:
        """
        Perform custom analysis with user-defined prompt
        
        Args:
            image_path: Path to the blueprint image
            custom_prompt: Custom prompt for analysis
            return_json: Whether to parse response as JSON
            
        Returns:
            Analysis result (JSON or raw text)
        """
        response = self.generate_response(image_path, custom_prompt)
        
        if return_json:
            try:
                return self._parse_json_response(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {"error": "Failed to parse response", "raw_response": response}
        return response
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from model response"""
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            json_str = response.strip()
        
        return json.loads(json_str)


class Qwen2VLInterface(HVACVLMInterface):
    """Qwen2-VL model interface"""
    
    def load_model(self):
        """Load Qwen2-VL model"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            logger.info(f"Loading Qwen2-VL model from {self.model_path}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # Load model with quantization if specified
            if self.quantization == "int8":
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    load_in_8bit=True
                )
            elif self.quantization == "int4":
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    load_in_4bit=True
                )
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map=self.device
                )
            
            self.model.eval()
            logger.info("Qwen2-VL model loaded successfully")
            
        except ImportError:
            logger.error("Qwen2-VL requires transformers>=4.35.0")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL model: {e}")
            raise
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> Any:
        """Preprocess image for Qwen2-VL"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return image
    
    def generate_response(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """Generate response using Qwen2-VL"""
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        image = self.preprocess_image(image)
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )
        
        # Decode response
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Remove prompt from response
        response = response.split("assistant\n")[-1] if "assistant" in response else response
        
        return response.strip()


class InternVLInterface(HVACVLMInterface):
    """InternVL model interface"""
    
    def load_model(self):
        """Load InternVL model"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading InternVL model from {self.model_path}")
            
            # Load tokenizer
            self.processor = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model
            if self.quantization == "int8":
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    load_in_8bit=True,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    trust_remote_code=True
                )
            
            self.model.eval()
            logger.info("InternVL model loaded successfully")
            
        except ImportError:
            logger.error("InternVL requires transformers with trust_remote_code=True")
            raise
        except Exception as e:
            logger.error(f"Failed to load InternVL model: {e}")
            raise
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> Any:
        """Preprocess image for InternVL"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return image
    
    def generate_response(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """Generate response using InternVL"""
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        image = self.preprocess_image(image)
        
        # Generate response using InternVL's chat method
        with torch.no_grad():
            response = self.model.chat(
                self.processor,
                image,
                prompt,
                generation_config={
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": temperature > 0
                }
            )
        
        return response.strip()


def create_hvac_vlm(
    model_type: str = "qwen2-vl",
    model_path: str = None,
    device: str = "cuda",
    quantization: Optional[str] = None
) -> HVACVLMInterface:
    """
    Factory function to create HVAC VLM interface
    
    Args:
        model_type: Type of VLM ('qwen2-vl' or 'internvl')
        model_path: Path to model checkpoint (or HuggingFace model ID)
        device: Device to run on ('cuda' or 'cpu')
        quantization: Quantization mode ('int8', 'int4', or None)
        
    Returns:
        HVAC VLM interface instance
    """
    if model_path is None:
        # Default model paths
        if model_type == "qwen2-vl":
            model_path = "Qwen/Qwen2-VL-7B-Instruct"
        elif model_type == "internvl":
            model_path = "OpenGVLab/InternVL-Chat-V1-5"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    if model_type == "qwen2-vl":
        return Qwen2VLInterface(model_path, device, quantization)
    elif model_type == "internvl":
        return InternVLInterface(model_path, device, quantization)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
