"""
BOM Generator and Connectivity Analyzer (Weeks 11-14 Implementation)
Generates Bill of Materials and analyzes system connectivity from symbol detections
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json

from .yoloplan_detector import SymbolDetection, Connection, Circuit, SymbolCategory

logger = logging.getLogger(__name__)


@dataclass
class BOMItem:
    """Bill of Materials item"""
    item_id: str
    description: str
    symbol_type: str
    category: str
    quantity: int
    unit: str = "EA"  # Each
    specifications: Dict[str, Any] = field(default_factory=dict)
    estimated_cost: Optional[float] = None
    notes: str = ""


class BOMGenerator:
    """
    Bill of Materials generation from symbol counts (Weeks 11-14)
    
    Features:
    - Symbol to equipment mapping
    - Quantity takeoff
    - Specification lookup
    - Cost estimation integration
    """
    
    # Equipment specifications database (simplified)
    EQUIPMENT_SPECS = {
        SymbolCategory.AHU: {
            'description': 'Air Handling Unit',
            'unit': 'EA',
            'specs': {'type': 'packaged', 'capacity': 'varies'}
        },
        SymbolCategory.FAN: {
            'description': 'Fan',
            'unit': 'EA',
            'specs': {'type': 'centrifugal/axial', 'cfm': 'varies'}
        },
        SymbolCategory.VAV: {
            'description': 'Variable Air Volume Box',
            'unit': 'EA',
            'specs': {'type': 'pressure_dependent', 'cfm_range': 'varies'}
        },
        SymbolCategory.FCU: {
            'description': 'Fan Coil Unit',
            'unit': 'EA',
            'specs': {'type': 'horizontal/vertical', 'capacity': 'varies'}
        },
        SymbolCategory.DAMPER: {
            'description': 'Damper',
            'unit': 'EA',
            'specs': {'type': 'motorized/manual', 'size': 'varies'}
        },
        SymbolCategory.DIFFUSER: {
            'description': 'Air Diffuser',
            'unit': 'EA',
            'specs': {'type': 'square/round', 'size': 'varies'}
        },
        SymbolCategory.THERMOSTAT: {
            'description': 'Thermostat',
            'unit': 'EA',
            'specs': {'type': 'programmable/smart', 'stages': 'varies'}
        },
    }
    
    def __init__(self, cost_database: Optional[Dict[str, float]] = None):
        """
        Initialize BOM generator
        
        Args:
            cost_database: Optional cost database for price estimation
        """
        self.cost_database = cost_database or {}
        logger.info("BOM Generator initialized")
    
    def generate_bom(self,
                    symbol_counts: Dict[str, int],
                    detections: List[SymbolDetection]) -> List[BOMItem]:
        """
        Generate BOM from detected symbols
        
        Args:
            symbol_counts: Count of each symbol type
            detections: Full detection results for analysis
            
        Returns:
            List of BOM items
        """
        bom_items = []
        item_counter = 1
        
        # Group symbols by category
        symbols_by_category = self._group_by_category(detections)
        
        for symbol_type, count in symbol_counts.items():
            # Find category for this symbol type
            category = self._find_category(symbol_type, detections)
            
            # Get specifications
            specs = self._get_specifications(category, symbol_type)
            
            # Estimate cost
            cost = self._estimate_cost(category, symbol_type, count)
            
            bom_item = BOMItem(
                item_id=f"BOM-{item_counter:04d}",
                description=specs['description'],
                symbol_type=symbol_type,
                category=category.value if category != SymbolCategory.UNKNOWN else 'unknown',
                quantity=count,
                unit=specs['unit'],
                specifications=specs['specs'],
                estimated_cost=cost,
                notes=f"Detected from blueprint analysis"
            )
            
            bom_items.append(bom_item)
            item_counter += 1
        
        logger.info(f"Generated BOM with {len(bom_items)} items, total quantity: {sum(item.quantity for item in bom_items)}")
        
        return bom_items
    
    def _group_by_category(self, detections: List[SymbolDetection]) -> Dict[SymbolCategory, List[SymbolDetection]]:
        """Group symbols by category"""
        grouped = {}
        for det in detections:
            if det.category not in grouped:
                grouped[det.category] = []
            grouped[det.category].append(det)
        return grouped
    
    def _find_category(self, symbol_type: str, detections: List[SymbolDetection]) -> SymbolCategory:
        """Find category for symbol type"""
        for det in detections:
            if det.symbol_type == symbol_type:
                return det.category
        return SymbolCategory.UNKNOWN
    
    def _get_specifications(self, category: SymbolCategory, symbol_type: str) -> Dict[str, Any]:
        """Get equipment specifications"""
        if category in self.EQUIPMENT_SPECS:
            return self.EQUIPMENT_SPECS[category]
        
        # Default specs
        return {
            'description': symbol_type.replace('_', ' ').title(),
            'unit': 'EA',
            'specs': {'type': 'standard'}
        }
    
    def _estimate_cost(self, category: SymbolCategory, symbol_type: str, quantity: int) -> Optional[float]:
        """Estimate cost from database"""
        if symbol_type in self.cost_database:
            unit_cost = self.cost_database[symbol_type]
            return unit_cost * quantity
        
        # Default estimates (placeholder)
        cost_estimates = {
            SymbolCategory.AHU: 5000.0,
            SymbolCategory.VAV: 800.0,
            SymbolCategory.FCU: 1200.0,
            SymbolCategory.FAN: 500.0,
            SymbolCategory.DAMPER: 150.0,
            SymbolCategory.DIFFUSER: 50.0,
            SymbolCategory.THERMOSTAT: 200.0,
        }
        
        if category in cost_estimates:
            return cost_estimates[category] * quantity
        
        return None
    
    def export_bom(self,
                  bom_items: List[BOMItem],
                  format: str = 'csv',
                  output_path: Optional[str] = None) -> str:
        """
        Export BOM to file
        
        Args:
            bom_items: List of BOM items
            format: Export format ('csv', 'excel', 'json')
            output_path: Output file path
            
        Returns:
            File path or string
        """
        if format == 'csv':
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                'item_id', 'description', 'symbol_type', 'category',
                'quantity', 'unit', 'estimated_cost', 'notes'
            ])
            writer.writeheader()
            
            for item in bom_items:
                writer.writerow({
                    'item_id': item.item_id,
                    'description': item.description,
                    'symbol_type': item.symbol_type,
                    'category': item.category,
                    'quantity': item.quantity,
                    'unit': item.unit,
                    'estimated_cost': item.estimated_cost or 'N/A',
                    'notes': item.notes
                })
            
            csv_str = output.getvalue()
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(csv_str)
                return output_path
            
            return csv_str
        
        elif format == 'json':
            export_data = [
                {
                    'item_id': item.item_id,
                    'description': item.description,
                    'symbol_type': item.symbol_type,
                    'category': item.category,
                    'quantity': item.quantity,
                    'unit': item.unit,
                    'specifications': item.specifications,
                    'estimated_cost': item.estimated_cost,
                    'notes': item.notes
                }
                for item in bom_items
            ]
            
            json_str = json.dumps(export_data, indent=2)
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(json_str)
                return output_path
            
            return json_str
        
        else:
            raise ValueError(f"Unsupported format: {format}")


class ConnectivityAnalyzer:
    """
    Netlist generation and connectivity analysis for HVAC systems (Weeks 11-14)
    
    Analyzes:
    - Duct connections
    - Pipe routing
    - Equipment relationships
    - Zone assignments
    """
    
    def __init__(self):
        self.proximity_threshold = 150  # pixels
        self.alignment_threshold = 20  # pixels
        logger.info("Connectivity Analyzer initialized")
    
    def generate_netlist(self,
                        detections: List[SymbolDetection],
                        image_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Generate system connectivity netlist
        
        Args:
            detections: List of detected symbols
            image_size: Optional image dimensions
            
        Returns:
            Netlist with connections, circuits, zones
        """
        # Build connection graph
        graph = self._build_connection_graph(detections)
        
        # Identify connections
        connections = self._identify_connections(detections, graph)
        
        # Identify circuits/zones
        circuits = self._identify_circuits(graph, detections)
        
        # Analyze system hierarchy
        hierarchy = self._analyze_hierarchy(graph, detections)
        
        netlist = {
            'connections': connections,
            'circuits': circuits,
            'hierarchy': hierarchy,
            'graph_stats': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'connected_components': nx.number_connected_components(graph.to_undirected())
            }
        }
        
        logger.info(f"Generated netlist with {len(connections)} connections and {len(circuits)} circuits")
        
        return netlist
    
    def _build_connection_graph(self, detections: List[SymbolDetection]) -> nx.DiGraph:
        """Build directed graph of symbol connections"""
        graph = nx.DiGraph()
        
        # Add nodes
        for det in detections:
            graph.add_node(det.id, 
                          symbol_type=det.symbol_type,
                          category=det.category.value,
                          center=det.center)
        
        # Add edges based on proximity and type compatibility
        for det1 in detections:
            for det2 in detections:
                if det1.id == det2.id:
                    continue
                
                # Check proximity
                distance = self._compute_distance(det1.center, det2.center)
                
                if distance < self.proximity_threshold:
                    # Check if connection makes sense
                    if self._is_compatible_connection(det1, det2):
                        connection_type = self._infer_connection_type(det1, det2)
                        graph.add_edge(det1.id, det2.id,
                                     connection_type=connection_type,
                                     distance=distance)
        
        return graph
    
    def _identify_connections(self,
                             detections: List[SymbolDetection],
                             graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """Identify explicit connections"""
        connections = []
        
        for edge in graph.edges(data=True):
            from_id, to_id, data = edge
            
            # Find detection objects
            from_det = next((d for d in detections if d.id == from_id), None)
            to_det = next((d for d in detections if d.id == to_id), None)
            
            if from_det and to_det:
                connections.append({
                    'from': {
                        'id': from_det.id,
                        'type': from_det.symbol_type,
                        'category': from_det.category.value
                    },
                    'to': {
                        'id': to_det.id,
                        'type': to_det.symbol_type,
                        'category': to_det.category.value
                    },
                    'connection_type': data.get('connection_type', 'unknown'),
                    'distance': data.get('distance', 0)
                })
        
        return connections
    
    def _identify_circuits(self,
                          graph: nx.DiGraph,
                          detections: List[SymbolDetection]) -> List[Dict[str, Any]]:
        """Identify circuits/zones in the system"""
        circuits = []
        
        # Find connected components (undirected)
        undirected = graph.to_undirected()
        components = list(nx.connected_components(undirected))
        
        for idx, component in enumerate(components):
            if len(component) < 2:
                continue
            
            # Analyze circuit type
            circuit_type = self._determine_circuit_type(component, detections)
            
            circuits.append({
                'circuit_id': idx + 1,
                'circuit_type': circuit_type,
                'symbol_ids': list(component),
                'num_symbols': len(component)
            })
        
        return circuits
    
    def _analyze_hierarchy(self,
                          graph: nx.DiGraph,
                          detections: List[SymbolDetection]) -> Dict[str, Any]:
        """Analyze system hierarchy (supply → distribution → terminal)"""
        hierarchy = {
            'primary_equipment': [],
            'distribution': [],
            'terminal_units': []
        }
        
        # Classify by category
        for det in detections:
            if det.category in [SymbolCategory.AHU, SymbolCategory.CHILLER, 
                               SymbolCategory.BOILER, SymbolCategory.FAN]:
                hierarchy['primary_equipment'].append({
                    'id': det.id,
                    'type': det.symbol_type,
                    'category': det.category.value
                })
            elif det.category in [SymbolCategory.VAV, SymbolCategory.FCU,
                                 SymbolCategory.DAMPER]:
                hierarchy['distribution'].append({
                    'id': det.id,
                    'type': det.symbol_type,
                    'category': det.category.value
                })
            elif det.category in [SymbolCategory.DIFFUSER, SymbolCategory.GRILLE,
                                 SymbolCategory.REGISTER]:
                hierarchy['terminal_units'].append({
                    'id': det.id,
                    'type': det.symbol_type,
                    'category': det.category.value
                })
        
        return hierarchy
    
    def _is_compatible_connection(self, det1: SymbolDetection, det2: SymbolDetection) -> bool:
        """Check if two symbols can be connected"""
        # Equipment can connect to distribution
        if det1.category in [SymbolCategory.AHU, SymbolCategory.FAN]:
            if det2.category in [SymbolCategory.VAV, SymbolCategory.DAMPER, SymbolCategory.SUPPLY_DUCT]:
                return True
        
        # Distribution can connect to terminals
        if det1.category in [SymbolCategory.VAV, SymbolCategory.FCU]:
            if det2.category in [SymbolCategory.DIFFUSER, SymbolCategory.GRILLE]:
                return True
        
        # Sensors connect to equipment/controls
        if det1.category in [SymbolCategory.TEMPERATURE_SENSOR, SymbolCategory.PRESSURE_SENSOR]:
            if det2.category in [SymbolCategory.CONTROLLER, SymbolCategory.THERMOSTAT]:
                return True
        
        return False
    
    def _infer_connection_type(self, det1: SymbolDetection, det2: SymbolDetection) -> str:
        """Infer type of connection"""
        if 'supply' in det1.symbol_type.lower() or 'supply' in det2.symbol_type.lower():
            return 'supply'
        elif 'return' in det1.symbol_type.lower() or 'return' in det2.symbol_type.lower():
            return 'return'
        elif 'exhaust' in det1.symbol_type.lower():
            return 'exhaust'
        elif det1.category in [SymbolCategory.TEMPERATURE_SENSOR, SymbolCategory.THERMOSTAT]:
            return 'control'
        
        return 'general'
    
    def _determine_circuit_type(self, component: set, detections: List[SymbolDetection]) -> str:
        """Determine circuit type from component symbols"""
        categories = set()
        for symbol_id in component:
            det = next((d for d in detections if d.id == symbol_id), None)
            if det:
                categories.add(det.category)
        
        if any(cat in categories for cat in [SymbolCategory.AHU, SymbolCategory.FAN]):
            return 'hvac_system'
        elif any(cat in categories for cat in [SymbolCategory.TEMPERATURE_SENSOR, SymbolCategory.THERMOSTAT]):
            return 'control_loop'
        
        return 'unknown'
    
    def _compute_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Compute Euclidean distance"""
        import numpy as np
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def visualize_netlist(self, netlist: Dict[str, Any], output_path: str = 'netlist.png'):
        """
        Visualize connectivity as graph (optional)
        
        Args:
            netlist: Generated netlist
            output_path: Output file path
        """
        try:
            import matplotlib.pyplot as plt
            
            graph = nx.DiGraph()
            
            # Add edges from connections
            for conn in netlist['connections']:
                from_id = conn['from']['id']
                to_id = conn['to']['id']
                conn_type = conn['connection_type']
                
                graph.add_edge(from_id, to_id, connection_type=conn_type)
            
            # Draw
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, with_labels=True, node_color='lightblue',
                   node_size=500, font_size=8, arrows=True)
            
            plt.title("HVAC System Connectivity")
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Saved netlist visualization to {output_path}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")


# Factory functions

def create_bom_generator(cost_database: Optional[Dict[str, float]] = None) -> BOMGenerator:
    """Create BOM generator instance"""
    return BOMGenerator(cost_database=cost_database)


def create_connectivity_analyzer() -> ConnectivityAnalyzer:
    """Create connectivity analyzer instance"""
    return ConnectivityAnalyzer()
