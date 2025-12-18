"""
Configuration Utilities for HVAC YOLO Training

This module provides utilities for generating, validating, and managing
training configurations for YOLO11 segmentation models.

Usage:
    # Generate default config
    python config_utils.py generate --output training_config.yaml
    
    # Validate existing config
    python config_utils.py validate --config training_config.yaml
    
    # Generate config for specific use case
    python config_utils.py generate --preset small_dataset --output config.yaml
"""

import yaml
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple
import sys


class ConfigValidator:
    """Validates training configuration files."""
    
    REQUIRED_SECTIONS = ['paths', 'model', 'hardware', 'training', 'augmentation']
    
    VALID_OPTIMIZERS = ['Adam', 'AdamW', 'SGD', 'RMSProp']
    VALID_MODELS = ['yolo11n-seg.pt', 'yolo11s-seg.pt', 'yolo11m-seg.pt', 
                    'yolo11l-seg.pt', 'yolo11x-seg.pt']
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.errors = []
        self.warnings = []
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Validate configuration and return status, errors, warnings."""
        self._check_required_sections()
        self._validate_paths()
        self._validate_model()
        self._validate_hardware()
        self._validate_training()
        self._validate_augmentation()
        self._validate_consistency()
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _check_required_sections(self):
        """Check all required sections are present."""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.config:
                self.errors.append(f"Missing required section: {section}")
    
    def _validate_paths(self):
        """Validate paths configuration."""
        if 'paths' not in self.config:
            return
        
        paths = self.config['paths']
        required_paths = ['data_yaml', 'project_dir', 'run_name']
        
        for path_key in required_paths:
            if path_key not in paths:
                self.errors.append(f"Missing required path: {path_key}")
    
    def _validate_model(self):
        """Validate model configuration."""
        if 'model' not in self.config:
            return
        
        model = self.config['model']
        
        if 'architecture' not in model:
            self.errors.append("Missing model architecture")
        elif model['architecture'] not in self.VALID_MODELS:
            self.warnings.append(
                f"Unusual model architecture: {model['architecture']}. "
                f"Valid options: {', '.join(self.VALID_MODELS)}"
            )
    
    def _validate_hardware(self):
        """Validate hardware configuration."""
        if 'hardware' not in self.config:
            return
        
        hw = self.config['hardware']
        
        # Image size
        if 'imgsz' in hw:
            if hw['imgsz'] < 320 or hw['imgsz'] > 2048:
                self.warnings.append(
                    f"Unusual image size: {hw['imgsz']}. "
                    f"Recommended: 640-1280"
                )
            if hw['imgsz'] % 32 != 0:
                self.errors.append(
                    f"Image size must be divisible by 32, got {hw['imgsz']}"
                )
        
        # Batch size
        if 'batch' in hw:
            if hw['batch'] < 1:
                self.errors.append("Batch size must be >= 1")
            elif hw['batch'] > 32:
                self.warnings.append(
                    f"Large batch size {hw['batch']} may cause OOM"
                )
        
        # Workers
        if 'workers' in hw:
            if hw['workers'] < 0:
                self.errors.append("Workers must be >= 0")
            elif hw['workers'] > 16:
                self.warnings.append(
                    f"High worker count {hw['workers']} may be inefficient"
                )
    
    def _validate_training(self):
        """Validate training configuration."""
        if 'training' not in self.config:
            return
        
        train = self.config['training']
        
        # Epochs
        if 'epochs' in train:
            if train['epochs'] < 1:
                self.errors.append("Epochs must be >= 1")
            elif train['epochs'] < 50:
                self.warnings.append(
                    f"Low epoch count {train['epochs']} may undertrain"
                )
        
        # Learning rate
        if 'lr0' in train:
            if train['lr0'] <= 0:
                self.errors.append("Learning rate must be > 0")
            elif train['lr0'] > 0.1:
                self.warnings.append(
                    f"High learning rate {train['lr0']} may cause instability"
                )
        
        # Optimizer
        if 'optimizer' in train:
            if train['optimizer'] not in self.VALID_OPTIMIZERS:
                self.warnings.append(
                    f"Unusual optimizer: {train['optimizer']}. "
                    f"Valid options: {', '.join(self.VALID_OPTIMIZERS)}"
                )
        
        # Patience
        if 'patience' in train and 'epochs' in train:
            if train['patience'] >= train['epochs']:
                self.warnings.append(
                    "Patience >= epochs, early stopping won't trigger"
                )
    
    def _validate_augmentation(self):
        """Validate augmentation configuration."""
        if 'augmentation' not in self.config:
            return
        
        aug = self.config['augmentation']
        
        # Check ranges for augmentation parameters
        range_checks = {
            'mosaic': (0.0, 1.0),
            'mixup': (0.0, 1.0),
            'copy_paste': (0.0, 1.0),
            'degrees': (0.0, 180.0),
            'translate': (0.0, 1.0),
            'scale': (0.0, 2.0),
            'fliplr': (0.0, 1.0),
            'flipud': (0.0, 1.0),
        }
        
        for param, (min_val, max_val) in range_checks.items():
            if param in aug:
                value = aug[param]
                if not (min_val <= value <= max_val):
                    self.warnings.append(
                        f"{param}={value} outside typical range "
                        f"[{min_val}, {max_val}]"
                    )
    
    def _validate_consistency(self):
        """Validate consistency across configuration."""
        # Check if close_mosaic is reasonable
        if 'training' in self.config and 'augmentation' in self.config:
            train = self.config['training']
            aug = self.config['augmentation']
            
            if 'close_mosaic' in train and 'epochs' in train:
                if train['close_mosaic'] >= train['epochs']:
                    self.warnings.append(
                        "close_mosaic >= epochs, mosaic won't be disabled"
                    )
                
                if aug.get('mosaic', 0) > 0 and train['close_mosaic'] == 0:
                    self.warnings.append(
                        "Mosaic enabled but close_mosaic=0 (will run entire training)"
                    )


class ConfigGenerator:
    """Generates training configuration files."""
    
    PRESETS = {
        'default': {
            'description': 'Balanced configuration for medium datasets (500-2000 images)',
            'model': 'yolo11m-seg.pt',
            'imgsz': 1024,
            'batch': 4,
            'epochs': 100,
        },
        'small_dataset': {
            'description': 'Optimized for small datasets (<500 images)',
            'model': 'yolo11s-seg.pt',
            'imgsz': 1024,
            'batch': 4,
            'epochs': 100,
            'patience': 15,
            'copy_paste': 0.5,  # Higher augmentation
        },
        'large_dataset': {
            'description': 'Optimized for large datasets (>2000 images)',
            'model': 'yolo11l-seg.pt',
            'imgsz': 1280,
            'batch': 8,
            'epochs': 150,
            'patience': 30,
        },
        'fast_training': {
            'description': 'Quick training for experiments',
            'model': 'yolo11s-seg.pt',
            'imgsz': 640,
            'batch': 8,
            'epochs': 50,
            'close_mosaic': 5,
        },
        'high_accuracy': {
            'description': 'Maximum accuracy (slow training)',
            'model': 'yolo11l-seg.pt',
            'imgsz': 1280,
            'batch': 4,
            'epochs': 200,
            'patience': 40,
            'lr0': 0.0005,  # Lower LR for stability
        },
    }
    
    @staticmethod
    def generate(preset: str = 'default', **overrides) -> Dict[str, Any]:
        """Generate configuration from preset with optional overrides."""
        if preset not in ConfigGenerator.PRESETS:
            raise ValueError(
                f"Unknown preset: {preset}. "
                f"Available: {', '.join(ConfigGenerator.PRESETS.keys())}"
            )
        
        preset_config = ConfigGenerator.PRESETS[preset].copy()
        description = preset_config.pop('description')
        
        # Apply overrides
        preset_config.update(overrides)
        
        # Build full configuration
        config = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'preset': preset,
                'description': description,
            },
            'paths': {
                'data_yaml': '/content/hvac_config.yaml',
                'project_dir': '/content/drive/MyDrive/hvac_detection_project/runs/segment',
                'run_name': f'hvac_{preset}_{datetime.now().strftime("%Y%m%d_%H%M")}',
            },
            'model': {
                'architecture': preset_config.get('model', 'yolo11m-seg.pt'),
                'pretrained': True,
                'freeze_layers': None,
            },
            'hardware': {
                'imgsz': preset_config.get('imgsz', 1024),
                'batch': preset_config.get('batch', 4),
                'workers': 2,
                'cache': False,
                'amp': True,
                'device': 0,
            },
            'training': {
                'epochs': preset_config.get('epochs', 100),
                'patience': preset_config.get('patience', 20),
                'save_period': 5,
                'close_mosaic': preset_config.get('close_mosaic', 15),
                'optimizer': 'AdamW',
                'lr0': preset_config.get('lr0', 0.001),
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
            },
            'augmentation': {
                'augment': True,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': preset_config.get('copy_paste', 0.3),
                'degrees': 10.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'fliplr': 0.5,
                'flipud': 0.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'use_albumentations': False,
                'albumentations_p': 0.5,
            },
            'loss_weights': {
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'seg': 1.0,
            },
            'validation': {
                'val': True,
                'plots': True,
                'save_json': True,
                'save_hybrid': True,
                'conf': 0.001,
                'iou': 0.6,
                'max_det': 300,
            },
            'logging': {
                'verbose': True,
                'tensorboard': True,
                'exist_ok': True,
            },
        }
        
        return config


def main():
    parser = argparse.ArgumentParser(
        description='Configuration utilities for HVAC YOLO training'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate configuration file')
    gen_parser.add_argument(
        '--preset',
        choices=list(ConfigGenerator.PRESETS.keys()),
        default='default',
        help='Configuration preset'
    )
    gen_parser.add_argument(
        '--output', '-o',
        default='training_config.yaml',
        help='Output file path'
    )
    gen_parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available presets'
    )
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate configuration file')
    val_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Configuration file to validate'
    )
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        if args.list_presets:
            print("Available presets:\n")
            for name, preset in ConfigGenerator.PRESETS.items():
                print(f"  {name}:")
                print(f"    {preset['description']}")
                print()
            return
        
        print(f"Generating configuration with preset '{args.preset}'...")
        config = ConfigGenerator.generate(preset=args.preset)
        
        with open(args.output, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Configuration saved to: {args.output}")
        print(f"\nPreset: {args.preset}")
        print(f"Description: {config['metadata']['description']}")
        print(f"\nKey parameters:")
        print(f"  Model: {config['model']['architecture']}")
        print(f"  Image size: {config['hardware']['imgsz']}")
        print(f"  Batch size: {config['hardware']['batch']}")
        print(f"  Epochs: {config['training']['epochs']}")
        
    elif args.command == 'validate':
        print(f"Validating configuration: {args.config}...")
        
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ Error: Configuration file not found: {args.config}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"❌ Error: Invalid YAML syntax in configuration file")
            print(f"   {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            sys.exit(1)
        
        validator = ConfigValidator(config)
        is_valid, errors, warnings = validator.validate()
        
        if errors:
            print("\n❌ Validation Errors:")
            for error in errors:
                print(f"  • {error}")
        
        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  • {warning}")
        
        if is_valid:
            print("\n✅ Configuration is valid!")
            if warnings:
                print("   (but has warnings - review above)")
        else:
            print("\n❌ Configuration is invalid. Fix errors above.")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
