"""
HVAC System Graph Builder

Constructs graph representation of HVAC systems using NetworkX
for advanced system analysis and flow path validation.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import logging

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available. System graph analysis will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class SystemNode:
    """Represents a node in the HVAC system graph"""
    node_id: str
    node_type: str  # "equipment", "duct", "diffuser", "damper", "junction"
    location: Tuple[float, float]
    attributes: Dict[str, Any]


@dataclass
class SystemEdge:
    """Represents a connection between nodes"""
    source_id: str
    target_id: str
    edge_type: str  # "duct_connection", "control_signal", "airflow"
    weight: float = 1.0  # Edge weight for pathfinding
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class SystemGraphBuilder:
    """
    HVAC System Graph Builder
    
    Builds NetworkX graph representation of HVAC systems for:
    - Flow path analysis
    - System connectivity validation
    - Shortest path calculations
    - System topology analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.graph: Optional[Any] = None
        
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()  # Directed graph for airflow direction
        else:
            self.logger.warning(
                "NetworkX not available. Install with: pip install networkx"
            )
    
    def add_node(self, node: SystemNode):
        """
        Add a node to the system graph
        
        Args:
            node: SystemNode to add
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return
        
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type,
            location=node.location,
            **node.attributes
        )
        
        self.logger.debug(f"Added node: {node.node_id} ({node.node_type})")
    
    def add_edge(self, edge: SystemEdge):
        """
        Add an edge to the system graph
        
        Args:
            edge: SystemEdge to add
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return
        
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type,
            weight=edge.weight,
            **edge.attributes
        )
        
        self.logger.debug(
            f"Added edge: {edge.source_id} -> {edge.target_id} ({edge.edge_type})"
        )
    
    def build_graph_from_components(
        self,
        components: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> bool:
        """
        Build graph from component and relationship data
        
        Args:
            components: List of component dictionaries
            relationships: List of relationship dictionaries
            
        Returns:
            True if graph built successfully
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return False
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes
        for component in components:
            node = SystemNode(
                node_id=component["id"],
                node_type=component.get("type", "unknown"),
                location=tuple(component.get("location", (0, 0))),
                attributes=component.get("attributes", {})
            )
            self.add_node(node)
        
        # Add edges
        for relationship in relationships:
            edge = SystemEdge(
                source_id=relationship["source_id"],
                target_id=relationship["target_id"],
                edge_type=relationship.get("type", "connection"),
                weight=relationship.get("weight", 1.0),
                attributes=relationship.get("attributes", {})
            )
            self.add_edge(edge)
        
        self.logger.info(
            f"Built system graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        
        return True
    
    def find_airflow_path(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[List[str]]:
        """
        Find airflow path between two components
        
        Args:
            source_id: Source component ID
            target_id: Target component ID
            
        Returns:
            List of node IDs in path, or None if no path exists
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return None
        
        try:
            path = nx.shortest_path(
                self.graph,
                source=source_id,
                target=target_id,
                weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            self.logger.debug(
                f"No path found from {source_id} to {target_id}"
            )
            return None
        except nx.NodeNotFound:
            self.logger.warning(
                f"Node not found: {source_id} or {target_id}"
            )
            return None
    
    def get_connected_components(self) -> List[Set[str]]:
        """
        Get weakly connected components in the system
        
        Returns:
            List of sets, where each set contains node IDs in a component
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return []
        
        return [
            set(component)
            for component in nx.weakly_connected_components(self.graph)
        ]
    
    def detect_isolated_components(self) -> List[str]:
        """
        Detect components that are isolated (not connected to main system)
        
        Returns:
            List of isolated component IDs
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return []
        
        components = self.get_connected_components()
        
        if not components:
            return []
        
        # Find largest component (assumed to be main system)
        largest_component = max(components, key=len)
        
        # All other components are isolated
        isolated = []
        for component in components:
            if component != largest_component:
                isolated.extend(list(component))
        
        return isolated
    
    def calculate_system_metrics(self) -> Dict[str, Any]:
        """
        Calculate system-level metrics
        
        Returns:
            Dictionary with system metrics
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return {
                "networkx_available": False,
                "message": "NetworkX not available"
            }
        
        metrics = {
            "networkx_available": True,
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "connected_components": len(list(nx.weakly_connected_components(self.graph))),
            # For directed graphs: average degree = (total in-degree + total out-degree) / nodes
            "average_degree": (2 * self.graph.number_of_edges()) / max(self.graph.number_of_nodes(), 1),
        }
        
        # Calculate density (only for graphs with nodes)
        if self.graph.number_of_nodes() > 1:
            metrics["density"] = nx.density(self.graph)
        else:
            metrics["density"] = 0.0
        
        # Identify hub nodes (highly connected)
        if self.graph.number_of_nodes() > 0:
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 0
            metrics["hub_nodes"] = [
                node for node, degree in degrees.items()
                if degree >= max_degree * 0.7  # Nodes with 70%+ of max degree
            ]
        else:
            metrics["hub_nodes"] = []
        
        return metrics
    
    def validate_system_connectivity(self) -> Dict[str, Any]:
        """
        Validate overall system connectivity
        
        Returns:
            Dictionary with connectivity validation results
        """
        violations = []
        
        if not NETWORKX_AVAILABLE or self.graph is None:
            return {
                "is_valid": False,
                "violations": [{
                    "severity": "WARNING",
                    "description": "System connectivity analysis requires NetworkX"
                }]
            }
        
        # Check for isolated components
        isolated = self.detect_isolated_components()
        
        if isolated:
            violations.append({
                "severity": "WARNING",
                "description": (
                    f"Found {len(isolated)} isolated components not connected "
                    f"to main HVAC system"
                ),
                "remediation": (
                    "Verify that all components are properly connected. "
                    "Isolated components may not receive air supply."
                ),
                "isolated_component_ids": isolated,
                "confidence": 0.80
            })
        
        # Check for nodes with no incoming or outgoing edges
        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            
            if in_degree == 0 and out_degree == 0:
                violations.append({
                    "severity": "WARNING",
                    "description": (
                        f"Component {node} has no connections to system"
                    ),
                    "remediation": "Connect component to HVAC system",
                    "confidence": 0.85
                })
        
        metrics = self.calculate_system_metrics()
        
        return {
            "is_valid": len([v for v in violations if v["severity"] in ["CRITICAL", "WARNING"]]) == 0,
            "violations": violations,
            "metrics": metrics,
            "summary": {
                "total_components": metrics["node_count"],
                "isolated_components": len(isolated),
                "connectivity_score": 100.0 - (len(isolated) / max(metrics["node_count"], 1) * 100.0)
            }
        }
    
    def export_graph_data(self) -> Dict[str, Any]:
        """
        Export graph data for visualization
        
        Returns:
            Dictionary with nodes and edges for visualization
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return {"nodes": [], "edges": []}
        
        nodes = [
            {
                "id": node,
                **self.graph.nodes[node]
            }
            for node in self.graph.nodes()
        ]
        
        edges = [
            {
                "source": edge[0],
                "target": edge[1],
                **self.graph.edges[edge]
            }
            for edge in self.graph.edges()
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": self.calculate_system_metrics()
        }
