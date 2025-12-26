"""
Supervised Fine-Tuning (SFT) for HVAC VLM

Implements supervised fine-tuning pipeline to transform general VLM
into HVAC-specific expert model.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class HVACVLMDataset(Dataset):
    """Dataset for HVAC VLM training"""
    
    def __init__(
        self,
        data_path: str,
        processor: Any,
        max_length: int = 2048
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to training data directory
            processor: Model processor for tokenization
            max_length: Maximum sequence length
        """
        self.data_path = Path(data_path)
        self.processor = processor
        self.max_length = max_length
        
        # Load data manifest
        import json
        manifest_path = self.data_path / "dataset_manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        
        self.samples = self.manifest["samples"]
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get training sample"""
        sample = self.samples[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Load annotations
        import json
        annotation_path = self.data_path / "annotations" / f"{sample['image_id']}.json"
        with open(annotation_path) as f:
            annotation_data = json.load(f)
        
        # Create prompt and response
        prompt = annotation_data["prompts"][0]["text"]
        
        # Create structured response from annotations
        response = self._create_response(annotation_data["annotations"])
        
        # Process inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        
        # Squeeze batch dimension
        return {k: v.squeeze(0) for k, v in inputs.items()}
    
    def _create_response(self, annotations: List[Dict]) -> str:
        """Create structured JSON response from annotations"""
        import json
        
        components = []
        for ann in annotations:
            components.append({
                "type": ann["component_type"],
                "bbox": ann["bbox"],
                "attributes": ann["attributes"]
            })
        
        response = {
            "components": components
        }
        
        return json.dumps(response, indent=2)


class SupervisedFinetuner:
    """Supervised fine-tuning trainer for HVAC VLM"""
    
    def __init__(
        self,
        model: Any,
        processor: Any,
        output_dir: str = "checkpoints/hvac_vlm_sft",
        use_lora: bool = True,
        lora_config: Optional[Dict] = None
    ):
        """
        Initialize SFT trainer
        
        Args:
            model: Base VLM model
            processor: Model processor
            output_dir: Directory to save checkpoints
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_config: LoRA configuration parameters
        """
        self.model = model
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_lora = use_lora
        
        # Apply LoRA if enabled
        if use_lora:
            self._apply_lora(lora_config or {})
    
    def _apply_lora(self, config: Dict):
        """Apply LoRA to model"""
        lora_config = LoraConfig(
            r=config.get("r", 16),
            lora_alpha=config.get("alpha", 32),
            target_modules=config.get(
                "target_modules",
                ["q_proj", "v_proj", "k_proj", "o_proj"]
            ),
            lora_dropout=config.get("dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA applied to model")
        logger.info(f"Trainable parameters: {self.model.print_trainable_parameters()}")
    
    def train(
        self,
        train_data_path: str,
        eval_data_path: Optional[str] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 8,
        save_steps: int = 5000,
        eval_steps: int = 1000,
        logging_steps: int = 100,
        max_length: int = 2048
    ):
        """
        Train model with supervised fine-tuning
        
        Args:
            train_data_path: Path to training data
            eval_data_path: Path to evaluation data (optional)
            num_epochs: Number of training epochs
            batch_size: Training batch size per device
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            gradient_accumulation_steps: Gradient accumulation steps
            save_steps: Steps between checkpoint saves
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
            max_length: Maximum sequence length
        """
        logger.info("Starting supervised fine-tuning...")
        
        # Create datasets
        train_dataset = HVACVLMDataset(
            train_data_path,
            self.processor,
            max_length
        )
        
        eval_dataset = None
        if eval_data_path:
            eval_dataset = HVACVLMDataset(
                eval_data_path,
                self.processor,
                max_length
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to=["tensorboard"],
            logging_dir=str(self.output_dir / "logs")
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor
        )
        
        # Train
        logger.info(f"Training on {len(train_dataset)} samples")
        if eval_dataset:
            logger.info(f"Evaluating on {len(eval_dataset)} samples")
        
        train_result = trainer.train()
        
        # Save final model
        self.save_model(str(self.output_dir / "final"))
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)
        
        logger.info("Training complete!")
        logger.info(f"Metrics: {metrics}")
        
        return metrics
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.use_lora:
            # Save LoRA weights
            self.model.save_pretrained(str(save_path))
        else:
            # Save full model
            self.model.save_pretrained(str(save_path))
        
        self.processor.save_pretrained(str(save_path))
        logger.info(f"Model saved to {save_path}")


def create_sft_trainer(
    model_type: str = "qwen2-vl",
    base_model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
    output_dir: str = "checkpoints/hvac_vlm_sft",
    use_lora: bool = True,
    device: str = "cuda"
) -> SupervisedFinetuner:
    """
    Create SFT trainer with initialized model
    
    Args:
        model_type: Type of VLM ('qwen2-vl' or 'internvl')
        base_model_path: Path to base model
        output_dir: Output directory for checkpoints
        use_lora: Whether to use LoRA
        device: Device to train on
        
    Returns:
        Initialized SFT trainer
    """
    from ..model_interface import create_hvac_vlm
    
    # Load base model
    vlm = create_hvac_vlm(
        model_type=model_type,
        model_path=base_model_path,
        device=device
    )
    vlm.load_model()
    
    # Create trainer
    trainer = SupervisedFinetuner(
        model=vlm.model,
        processor=vlm.processor,
        output_dir=output_dir,
        use_lora=use_lora
    )
    
    return trainer
