"""
Synthetic HVAC Drawing Generator

Generates synthetic HVAC drawings with automatic annotations for VLM training.
Uses SVG/DXF generation to create perfectly labeled training data.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data_schema import (
    HVACComponentType,
    ComponentAttributes,
    ComponentAnnotation,
    DrawingMetadata,
    HVACTrainingExample,
    RelationshipType,
    DUCTWORK_TYPES,
    EQUIPMENT_TYPES,
    CONTROL_TYPES
)


class SyntheticDataGenerator:
    """Generate synthetic HVAC drawings for training"""
    
    def __init__(
        self,
        output_dir: str = "datasets/synthetic",
        image_size: Tuple[int, int] = (2048, 2048),
        dpi: int = 300
    ):
        """
        Initialize synthetic data generator
        
        Args:
            output_dir: Directory to save generated data
            image_size: Size of generated images (width, height)
            dpi: DPI resolution for images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.dpi = dpi
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        
    def generate_simple_supply_system(
        self,
        num_components: int = 10,
        include_labels: bool = True
    ) -> HVACTrainingExample:
        """
        Generate a simple supply air system
        
        Args:
            num_components: Number of components to generate
            include_labels: Whether to include text labels
            
        Returns:
            Complete training example with image and annotations
        """
        image_id = f"synthetic_supply_{random.randint(1000, 9999)}"
        
        # Create blank image (white background)
        img = Image.new("RGB", self.image_size, color="white")
        draw = ImageDraw.Draw(img)
        
        annotations = []
        
        # Generate AHU (Air Handling Unit) at the start
        ahu_x, ahu_y = 200, self.image_size[1] // 2
        ahu_w, ahu_h = 150, 100
        annotations.append(self._draw_ahu(draw, ahu_x, ahu_y, ahu_w, ahu_h))
        
        # Generate main supply duct from AHU
        duct_start_x = ahu_x + ahu_w
        duct_y = ahu_y + ahu_h // 2
        duct_width = 20
        duct_length = 400
        
        main_duct = self._draw_duct(
            draw,
            duct_start_x,
            duct_y - duct_width // 2,
            duct_length,
            duct_width,
            "12x10",
            2000,
            "SD-1"
        )
        annotations.append(main_duct)
        
        # Generate branches with VAVs and diffusers
        num_branches = min(num_components - 2, 4)  # -2 for AHU and main duct
        branch_spacing = duct_length // (num_branches + 1)
        
        for i in range(num_branches):
            branch_x = duct_start_x + (i + 1) * branch_spacing
            
            # VAV box
            vav_y = duct_y - 100
            vav = self._draw_vav(draw, branch_x - 30, vav_y, 60, 40, f"VAV-{i+101}")
            annotations.append(vav)
            
            # Duct from main to VAV
            branch_duct = self._draw_duct(
                draw,
                branch_x - 5,
                duct_y,
                5,
                90,
                "8x8",
                500,
                f"SD-{i+2}",
                vertical=True
            )
            annotations.append(branch_duct)
            
            # Diffusers below VAV
            diffuser = self._draw_diffuser(draw, branch_x - 20, vav_y - 60, 40, 40)
            annotations.append(diffuser)
        
        # Add relationships
        self._add_relationships(annotations)
        
        # Save image
        image_path = self.output_dir / "images" / f"{image_id}.png"
        img.save(image_path, dpi=(self.dpi, self.dpi))
        
        # Create training example
        metadata = DrawingMetadata(
            drawing_type="supply_air_plan",
            system_type="commercial",
            complexity="simple",
            resolution=f"{self.dpi}dpi"
        )
        
        prompts = [
            {
                "type": "component_detection",
                "text": "Identify all HVAC components in this supply air plan."
            },
            {
                "type": "relationship_analysis",
                "text": "Describe the airflow path from the AHU through the system."
            }
        ]
        
        example = HVACTrainingExample(
            image_id=image_id,
            image_path=str(image_path),
            metadata=metadata,
            annotations=annotations,
            prompts=prompts
        )
        
        # Save annotations
        self._save_annotations(example)
        
        return example
    
    def generate_dataset(
        self,
        num_samples: int = 100,
        complexity_distribution: Dict[str, float] = None
    ) -> List[HVACTrainingExample]:
        """
        Generate a complete dataset
        
        Args:
            num_samples: Number of samples to generate
            complexity_distribution: Distribution of complexity levels
            
        Returns:
            List of training examples
        """
        if complexity_distribution is None:
            complexity_distribution = {
                "simple": 0.4,
                "medium": 0.4,
                "complex": 0.2
            }
        
        examples = []
        
        for i in range(num_samples):
            # Select complexity
            complexity = random.choices(
                list(complexity_distribution.keys()),
                weights=list(complexity_distribution.values())
            )[0]
            
            # Generate based on complexity
            if complexity == "simple":
                num_components = random.randint(5, 10)
            elif complexity == "medium":
                num_components = random.randint(10, 20)
            else:
                num_components = random.randint(20, 40)
            
            example = self.generate_simple_supply_system(
                num_components=num_components
            )
            examples.append(example)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        # Save dataset manifest
        self._save_dataset_manifest(examples)
        
        return examples
    
    def _draw_ahu(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> ComponentAnnotation:
        """Draw Air Handling Unit"""
        # Draw rectangle for AHU
        draw.rectangle(
            [(x, y), (x + width, y + height)],
            outline="black",
            width=3
        )
        
        # Add label
        draw.text(
            (x + width // 2 - 20, y + height // 2),
            "AHU-1",
            fill="black"
        )
        
        return ComponentAnnotation(
            component_id=f"ahu_{random.randint(1000, 9999)}",
            component_type=HVACComponentType.AHU,
            bbox=(x, y, x + width, y + height),
            attributes=ComponentAttributes(
                designation="AHU-1",
                cfm=5000,
                capacity=100.0  # tons
            )
        )
    
    def _draw_duct(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        length: int,
        width: int,
        size: str,
        cfm: int,
        designation: str,
        vertical: bool = False
    ) -> ComponentAnnotation:
        """Draw duct section"""
        if vertical:
            # Vertical duct
            coords = [(x, y), (x + width, y + length)]
        else:
            # Horizontal duct
            coords = [(x, y), (x + length, y + width)]
        
        # Draw duct
        draw.rectangle(coords, outline="black", width=2)
        
        # Add centerline
        if vertical:
            center_x = x + width // 2
            draw.line([(center_x, y), (center_x, y + length)], fill="blue", width=1)
        else:
            center_y = y + width // 2
            draw.line([(x, center_y), (x + length, center_y)], fill="blue", width=1)
        
        # Add label
        label = f"{designation}\n{size}\n{cfm}CFM"
        if vertical:
            draw.text((x + width + 5, y + length // 2 - 20), label, fill="black")
        else:
            draw.text((x + length // 2 - 30, y - 30), label, fill="black")
        
        return ComponentAnnotation(
            component_id=f"duct_{random.randint(1000, 9999)}",
            component_type=HVACComponentType.SUPPLY_AIR_DUCT,
            bbox=(coords[0][0], coords[0][1], coords[1][0], coords[1][1]),
            attributes=ComponentAttributes(
                size=size,
                cfm=cfm,
                designation=designation,
                material="galvanized_steel"
            )
        )
    
    def _draw_vav(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        width: int,
        height: int,
        designation: str
    ) -> ComponentAnnotation:
        """Draw VAV box"""
        # Draw rectangle
        draw.rectangle(
            [(x, y), (x + width, y + height)],
            outline="black",
            fill="lightgray",
            width=2
        )
        
        # Add label
        draw.text((x + 5, y + height // 2 - 5), designation, fill="black")
        
        return ComponentAnnotation(
            component_id=f"vav_{random.randint(1000, 9999)}",
            component_type=HVACComponentType.VAV,
            bbox=(x, y, x + width, y + height),
            attributes=ComponentAttributes(
                designation=designation,
                cfm=random.randint(500, 2000),
                size=f"{random.choice([8, 10, 12])}\""
            )
        )
    
    def _draw_diffuser(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> ComponentAnnotation:
        """Draw diffuser"""
        # Draw X pattern for diffuser
        draw.line([(x, y), (x + width, y + height)], fill="black", width=2)
        draw.line([(x + width, y), (x, y + height)], fill="black", width=2)
        draw.rectangle([(x, y), (x + width, y + height)], outline="black", width=2)
        
        return ComponentAnnotation(
            component_id=f"diffuser_{random.randint(1000, 9999)}",
            component_type=HVACComponentType.DIFFUSER,
            bbox=(x, y, x + width, y + height),
            attributes=ComponentAttributes(
                size="12x12",
                cfm=random.randint(100, 400)
            )
        )
    
    def _add_relationships(self, annotations: List[ComponentAnnotation]):
        """Add relationships between components"""
        # Find AHU
        ahu = next((a for a in annotations if a.component_type == HVACComponentType.AHU), None)
        if not ahu:
            return
        
        # AHU supplies main ducts
        for ann in annotations:
            if ann.component_type == HVACComponentType.SUPPLY_AIR_DUCT:
                # Check if duct is near AHU (simple heuristic)
                if abs(ann.bbox[0] - ahu.bbox[2]) < 50:
                    ann.relationships.append({
                        "type": RelationshipType.SUPPLIES.value,
                        "source": ahu.component_id,
                        "target": ann.component_id
                    })
    
    def _save_annotations(self, example: HVACTrainingExample):
        """Save annotations to JSON file"""
        annotation_path = self.output_dir / "annotations" / f"{example.image_id}.json"
        
        # Convert to dict
        data = {
            "image_id": example.image_id,
            "image_path": example.image_path,
            "metadata": asdict(example.metadata),
            "annotations": [
                {
                    "component_id": ann.component_id,
                    "component_type": ann.component_type.value,
                    "bbox": ann.bbox,
                    "polygon": ann.polygon,
                    "attributes": asdict(ann.attributes),
                    "relationships": ann.relationships,
                    "confidence": ann.confidence
                }
                for ann in example.annotations
            ],
            "prompts": example.prompts
        }
        
        with open(annotation_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _save_dataset_manifest(self, examples: List[HVACTrainingExample]):
        """Save dataset manifest"""
        manifest_path = self.output_dir / "dataset_manifest.json"
        
        manifest = {
            "num_samples": len(examples),
            "image_size": self.image_size,
            "dpi": self.dpi,
            "samples": [
                {
                    "image_id": ex.image_id,
                    "image_path": ex.image_path,
                    "num_components": len(ex.annotations),
                    "drawing_type": ex.metadata.drawing_type,
                    "complexity": ex.metadata.complexity
                }
                for ex in examples
            ]
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\nDataset manifest saved to {manifest_path}")


def generate_training_dataset(
    output_dir: str = "datasets/synthetic_hvac_v1",
    num_samples: int = 1000,
    image_size: Tuple[int, int] = (2048, 2048)
) -> List[HVACTrainingExample]:
    """
    Generate a complete training dataset
    
    Args:
        output_dir: Directory to save dataset
        num_samples: Number of samples to generate
        image_size: Size of images
        
    Returns:
        List of training examples
    """
    generator = SyntheticDataGenerator(
        output_dir=output_dir,
        image_size=image_size
    )
    
    print(f"Generating {num_samples} synthetic HVAC drawings...")
    examples = generator.generate_dataset(num_samples)
    print(f"Dataset generation complete! Saved to {output_dir}")
    
    return examples
