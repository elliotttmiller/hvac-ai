"""
Pricing Engine for HVAC Quote Generation
Converts YOLO detection counts into detailed financial quotes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LineItem(BaseModel):
    """Individual line item in a quote"""
    category: str
    count: int
    unit_material_cost: float
    unit_labor_hours: float
    total_line_cost: float
    sku_name: Optional[str] = None
    unit: Optional[str] = "each"


class QuoteSummary(BaseModel):
    """Quote summary with totals"""
    subtotal_materials: float
    subtotal_labor: float
    total_cost: float
    final_price: float


class QuoteResponse(BaseModel):
    """Complete quote response"""
    quote_id: str
    currency: str = "USD"
    summary: QuoteSummary
    line_items: List[LineItem]


class QuoteSettings(BaseModel):
    """Quote calculation settings"""
    margin_percent: float = Field(default=20.0, ge=0, le=100)
    tax_rate: float = Field(default=0.0, ge=0, le=100)
    labor_hourly_rate: float = Field(default=85.0, gt=0)


class AnalysisData(BaseModel):
    """Analysis data from YOLO detection"""
    total_objects: int
    counts_by_category: Dict[str, int]


class QuoteRequest(BaseModel):
    """Request payload for quote generation"""
    project_id: str
    location: str
    analysis_data: AnalysisData
    settings: Optional[QuoteSettings] = None


class PricingEngine:
    """
    Pricing Engine that converts YOLO detection results into financial quotes.
    Applies regional multipliers and calculates materials + labor costs.
    """

    def __init__(self, catalog_path: Optional[Path] = None):
        """
        Initialize the pricing engine with a catalog.
        
        Args:
            catalog_path: Path to catalog.json file. If None, uses default location.
        """
        if catalog_path is None:
            catalog_path = Path(__file__).parent / "catalog.json"
        
        self.catalog_path = catalog_path
        self.catalog = self._load_catalog()
        logger.info(f"✅ Pricing Engine initialized with {len(self.catalog['components'])} component types")

    def _load_catalog(self) -> Dict:
        """Load pricing catalog from JSON file"""
        try:
            with open(self.catalog_path, 'r') as f:
                catalog = json.load(f)
            
            # Validate catalog structure
            if 'components' not in catalog:
                raise ValueError("Catalog missing 'components' key")
            if 'regional_multipliers' not in catalog:
                raise ValueError("Catalog missing 'regional_multipliers' key")
            if 'default_rates' not in catalog:
                raise ValueError("Catalog missing 'default_rates' key")
            
            return catalog
        except Exception as e:
            logger.error(f"Failed to load catalog from {self.catalog_path}: {e}")
            raise

    def _get_regional_multipliers(self, location: str) -> Dict[str, float]:
        """
        Get regional cost multipliers for a location.
        
        Args:
            location: Location string (e.g., "Chicago, IL")
            
        Returns:
            Dict with 'labor' and 'material' multipliers
        """
        multipliers = self.catalog['regional_multipliers']
        
        # Try exact match first
        if location in multipliers:
            return multipliers[location]
        
        # Try partial match (e.g., "Chicago" matches "Chicago, IL")
        location_lower = location.lower()
        for key in multipliers:
            if location_lower in key.lower() or key.lower() in location_lower:
                logger.info(f"Matched location '{location}' to '{key}'")
                return multipliers[key]
        
        # Fall back to default
        logger.info(f"No match for location '{location}', using default multipliers")
        return multipliers.get('default', {'labor': 1.0, 'material': 1.0})

    def _get_component_pricing(self, category: str) -> Dict:
        """
        Get pricing info for a component category.
        
        Args:
            category: Component category (e.g., "vav_box", "thermostat")
            
        Returns:
            Dict with material_cost, labor_hours, sku_name, etc.
        """
        components = self.catalog['components']
        
        # Try exact match
        if category in components:
            return components[category]
        
        # Try normalized match (replace spaces with underscores, lowercase)
        normalized = category.lower().replace(' ', '_').replace('-', '_')
        if normalized in components:
            return components[normalized]
        
        # Try partial match
        for key in components:
            if normalized in key or key in normalized:
                logger.info(f"Matched category '{category}' to '{key}'")
                return components[key]
        
        # Fall back to default "Miscellaneous" pricing
        logger.warning(f"No pricing found for category '{category}', using default")
        defaults = self.catalog['default_rates']
        return {
            'sku_name': f"Miscellaneous - {category}",
            'material_cost': defaults['material_cost'],
            'labor_hours': defaults['labor_hours'],
            'unit': 'each',
            'category': 'miscellaneous'
        }

    def _round_currency(self, value: float) -> float:
        """Round currency value to 2 decimal places"""
        return float(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    def generate_quote(self, request: QuoteRequest) -> QuoteResponse:
        """
        Generate a complete quote from analysis data.
        
        Args:
            request: QuoteRequest with project details and detection counts
            
        Returns:
            QuoteResponse with line items and totals
        """
        logger.info(f"🔄 Generating quote for project {request.project_id}")
        
        # Get settings with defaults
        settings = request.settings or QuoteSettings()
        
        # Get regional multipliers
        regional = self._get_regional_multipliers(request.location)
        labor_multiplier = regional['labor']
        material_multiplier = regional['material']
        
        logger.info(f"📍 Location: {request.location} (Labor: {labor_multiplier}x, Material: {material_multiplier}x)")
        
        # Calculate line items
        line_items: List[LineItem] = []
        subtotal_materials = 0.0
        subtotal_labor = 0.0
        
        for category, count in request.analysis_data.counts_by_category.items():
            if count <= 0:
                continue
            
            # Get pricing for this category
            pricing = self._get_component_pricing(category)
            
            # Calculate costs
            unit_material_cost = pricing['material_cost'] * material_multiplier
            unit_labor_hours = pricing['labor_hours']
            unit_labor_cost = unit_labor_hours * settings.labor_hourly_rate * labor_multiplier
            
            # Total for this line
            line_material_cost = unit_material_cost * count
            line_labor_cost = unit_labor_cost * count
            total_line_cost = line_material_cost + line_labor_cost
            
            # Add to subtotals
            subtotal_materials += line_material_cost
            subtotal_labor += line_labor_cost
            
            # Create line item
            line_items.append(LineItem(
                category=category,
                count=count,
                unit_material_cost=self._round_currency(unit_material_cost),
                unit_labor_hours=self._round_currency(unit_labor_hours),
                total_line_cost=self._round_currency(total_line_cost),
                sku_name=pricing.get('sku_name'),
                unit=pricing.get('unit', 'each')
            ))
        
        # Round subtotals
        subtotal_materials = self._round_currency(subtotal_materials)
        subtotal_labor = self._round_currency(subtotal_labor)
        total_cost = self._round_currency(subtotal_materials + subtotal_labor)
        
        # Apply margin
        margin_amount = total_cost * (settings.margin_percent / 100.0)
        final_price = self._round_currency(total_cost + margin_amount)
        
        logger.info(f"✅ Quote generated: Materials ${subtotal_materials}, Labor ${subtotal_labor}, Total ${final_price}")
        
        # Create response
        response = QuoteResponse(
            quote_id=f"Q-{request.project_id}",
            currency="USD",
            summary=QuoteSummary(
                subtotal_materials=subtotal_materials,
                subtotal_labor=subtotal_labor,
                total_cost=total_cost,
                final_price=final_price
            ),
            line_items=line_items
        )
        
        return response


def create_pricing_engine(catalog_path: Optional[Path] = None) -> PricingEngine:
    """Factory function to create a pricing engine"""
    return PricingEngine(catalog_path=catalog_path)
