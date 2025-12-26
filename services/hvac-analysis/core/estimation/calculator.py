"""
Estimation Engine Module
Material quantity calculations, labor estimation, and cost analysis
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MaterialItem:
    """Material item for estimation"""
    item_id: str
    description: str
    quantity: float
    unit: str
    unit_cost: float
    total_cost: float
    category: str

@dataclass
class LaborItem:
    """Labor item for estimation"""
    task: str
    hours: float
    hourly_rate: float
    total_cost: float
    crew_size: int = 1

class EstimationEngine:
    """
    Cost and labor estimation engine for HVAC projects
    """
    
    def __init__(self):
        self._load_pricing_data()
        
    def _load_pricing_data(self):
        """Load material and labor pricing data"""
        # In production, load from database with regional pricing
        
        self.material_pricing = {
            'hvac_unit': {
                '3_ton': {'cost': 2500, 'unit': 'each'},
                '5_ton': {'cost': 4200, 'unit': 'each'},
                '7.5_ton': {'cost': 6800, 'unit': 'each'},
                '10_ton': {'cost': 8500, 'unit': 'each'},
            },
            'duct': {
                '6_inch': {'cost': 8.50, 'unit': 'linear_foot'},
                '8_inch': {'cost': 11.20, 'unit': 'linear_foot'},
                '10_inch': {'cost': 14.80, 'unit': 'linear_foot'},
                '12_inch': {'cost': 18.50, 'unit': 'linear_foot'},
                '14_inch': {'cost': 22.30, 'unit': 'linear_foot'},
            },
            'vav_box': {
                'standard': {'cost': 1200, 'unit': 'each'},
                'with_reheat': {'cost': 1850, 'unit': 'each'},
            },
            'diffuser': {
                '2x2': {'cost': 45, 'unit': 'each'},
                '4x4': {'cost': 85, 'unit': 'each'},
            },
            'thermostat': {
                'programmable': {'cost': 125, 'unit': 'each'},
                'smart': {'cost': 250, 'unit': 'each'},
            },
            'damper': {
                'manual': {'cost': 180, 'unit': 'each'},
                'motorized': {'cost': 420, 'unit': 'each'},
            }
        }
        
        self.labor_rates = {
            'hvac_mechanic': 85.00,
            'apprentice': 45.00,
            'electrician': 95.00,
            'laborer': 35.00,
            'supervisor': 110.00
        }
        
        self.installation_hours = {
            'hvac_unit': {
                '3_ton': 12,
                '5_ton': 16,
                '7.5_ton': 20,
                '10_ton': 24,
            },
            'duct': 0.5,  # hours per linear foot
            'vav_box': 4,
            'diffuser': 1.5,
            'thermostat': 2,
            'damper': 3
        }
    
    def estimate_materials(self, components: List[Dict[str, Any]]) -> List[MaterialItem]:
        """
        Calculate material quantities and costs
        
        Args:
            components: List of detected components with specifications
            
        Returns:
            List of material items with costs
        """
        materials = []
        
        for comp in components:
            comp_type = comp.get('component_type', '')
            specs = comp.get('specifications', {})
            
            if comp_type == 'hvac_unit':
                capacity = specs.get('capacity_tons', 5)
                size_key = f'{capacity}_ton'
                
                if size_key in self.material_pricing['hvac_unit']:
                    pricing = self.material_pricing['hvac_unit'][size_key]
                    materials.append(MaterialItem(
                        item_id=f"HVAC_{comp.get('component_id')}",
                        description=f"{capacity} Ton HVAC Unit",
                        quantity=1,
                        unit='each',
                        unit_cost=pricing['cost'],
                        total_cost=pricing['cost'],
                        category='equipment'
                    ))
            
            elif comp_type == 'duct':
                diameter = specs.get('diameter_inches', 12)
                length = specs.get('length_feet', 10)
                size_key = f'{diameter}_inch'
                
                if size_key in self.material_pricing['duct']:
                    pricing = self.material_pricing['duct'][size_key]
                    materials.append(MaterialItem(
                        item_id=f"DUCT_{comp.get('component_id')}",
                        description=f"{diameter}\" Ductwork",
                        quantity=length,
                        unit='linear_foot',
                        unit_cost=pricing['cost'],
                        total_cost=pricing['cost'] * length,
                        category='ductwork'
                    ))
            
            elif comp_type == 'vav_box':
                pricing = self.material_pricing['vav_box']['standard']
                materials.append(MaterialItem(
                    item_id=f"VAV_{comp.get('component_id')}",
                    description="VAV Box with Controls",
                    quantity=1,
                    unit='each',
                    unit_cost=pricing['cost'],
                    total_cost=pricing['cost'],
                    category='equipment'
                ))
        
        return materials
    
    def estimate_labor(self, components: List[Dict[str, Any]]) -> List[LaborItem]:
        """
        Calculate labor hours and costs
        
        Args:
            components: List of detected components
            
        Returns:
            List of labor items with costs
        """
        labor_items = []
        mechanic_rate = self.labor_rates['hvac_mechanic']
        electrician_rate = self.labor_rates['electrician']
        
        for comp in components:
            comp_type = comp.get('component_type', '')
            specs = comp.get('specifications', {})
            
            if comp_type == 'hvac_unit':
                capacity = specs.get('capacity_tons', 5)
                size_key = f'{capacity}_ton'
                hours = self.installation_hours['hvac_unit'].get(size_key, 16)
                
                labor_items.append(LaborItem(
                    task=f"Install {capacity} Ton HVAC Unit",
                    hours=hours,
                    hourly_rate=mechanic_rate,
                    total_cost=hours * mechanic_rate,
                    crew_size=2
                ))
            
            elif comp_type == 'duct':
                length = specs.get('length_feet', 10)
                hours = length * self.installation_hours['duct']
                
                labor_items.append(LaborItem(
                    task=f"Install Ductwork ({length} LF)",
                    hours=hours,
                    hourly_rate=mechanic_rate,
                    total_cost=hours * mechanic_rate,
                    crew_size=2
                ))
            
            elif comp_type == 'thermostat':
                hours = self.installation_hours['thermostat']
                
                labor_items.append(LaborItem(
                    task="Install Thermostat",
                    hours=hours,
                    hourly_rate=electrician_rate,
                    total_cost=hours * electrician_rate,
                    crew_size=1
                ))
        
        # Add testing and commissioning
        total_install_hours = sum(item.hours for item in labor_items)
        commissioning_hours = max(8, total_install_hours * 0.1)
        
        labor_items.append(LaborItem(
            task="System Testing & Commissioning",
            hours=commissioning_hours,
            hourly_rate=mechanic_rate,
            total_cost=commissioning_hours * mechanic_rate,
            crew_size=1
        ))
        
        return labor_items
    
    def calculate_total_estimate(
        self,
        materials: List[MaterialItem],
        labor: List[LaborItem],
        location_multipliers: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate total project estimate
        
        Args:
            materials: List of material items
            labor: List of labor items
            location_multipliers: Regional cost adjustments
            
        Returns:
            Complete estimate breakdown
        """
        # Base costs
        material_subtotal = sum(m.total_cost for m in materials)
        labor_subtotal = sum(l.total_cost for l in labor)
        
        # Apply regional multipliers
        if location_multipliers:
            material_total = material_subtotal * location_multipliers.get('material', 1.0)
            labor_total = labor_subtotal * location_multipliers.get('labor', 1.0)
        else:
            material_total = material_subtotal
            labor_total = labor_subtotal
        
        # Additional costs
        markup = 0.15  # 15% markup
        contingency = 0.10  # 10% contingency
        
        subtotal = material_total + labor_total
        markup_amount = subtotal * markup
        contingency_amount = subtotal * contingency
        
        total = subtotal + markup_amount + contingency_amount
        
        return {
            'material_costs': {
                'subtotal': material_subtotal,
                'adjusted_total': material_total,
                'items': [
                    {
                        'description': m.description,
                        'quantity': m.quantity,
                        'unit': m.unit,
                        'unit_cost': m.unit_cost,
                        'total': m.total_cost
                    }
                    for m in materials
                ]
            },
            'labor_costs': {
                'subtotal': labor_subtotal,
                'adjusted_total': labor_total,
                'total_hours': sum(l.hours for l in labor),
                'items': [
                    {
                        'task': l.task,
                        'hours': l.hours,
                        'rate': l.hourly_rate,
                        'total': l.total_cost
                    }
                    for l in labor
                ]
            },
            'summary': {
                'subtotal': subtotal,
                'markup': markup_amount,
                'contingency': contingency_amount,
                'total': total
            },
            'adjustments': location_multipliers or {}
        }


def create_estimation_engine() -> EstimationEngine:
    """Factory function to create estimation engine"""
    return EstimationEngine()
