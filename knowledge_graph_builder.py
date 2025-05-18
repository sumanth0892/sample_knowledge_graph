"""
This is STEP 5 while building the knowledge graph, following
Step 1: PDF/Input data parsing
Step 2: Text preprocessing cleaning
Step 3: Entity extraction
Step 4: Relationship extraction
"""
import json
import random
import argparse
from collections import Counter
from tqdm import tqdm
from neo4j import GraphDatabase
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyvis.network import Network
from neo4j import GraphDatabase

class KnowledgeGraphBuilder:
    """
    A class to construct and visualize a knowledge graph from extracted entities and relationships.
    """
    
    def __init__(self, entities_dir=None, relationships_dir=None, output_dir=None):
        """
        Initialize the knowledge graph builder.
        
        Args:
            entities_dir: Directory containing entity JSON files
            relationships_dir: Directory containing relationship JSON files
            output_dir: Directory to save the knowledge graph files
        """
        self.entities_dir = Path(entities_dir) if entities_dir else None
        self.relationships_dir = Path(relationships_dir) if relationships_dir else None
        self.output_dir = Path(output_dir) if output_dir else Path('knowledge_graph')
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        # Initialize the graph
        self.graph = nx.DiGraph()
        
        # Color mapping for entity types
        self.entity_colors = {
            'PROTEIN_MODEL': '#FF5733',  # Red-orange
            'ORGANIZATION': '#33A8FF',   # Blue
            'BIOLOGICAL_ENTITY': '#33FF57',  # Green
            'TASK': '#A833FF',  # Purple
            'FIELD': '#FFBB33',  # Orange
            'TECHNIQUE': '#33FFC1',  # Teal
            'APPROACH': '#FF33A8',  # Pink
            'ML_MODEL': '#8CFF33',  # Lime
            'ML_COMPONENT': '#33FFE0',  # Cyan
            'PHYSICAL_PROPERTY': '#FF9C33',  # Light orange
            'DATABASE': '#33FFA8',  # Mint
            'REPRESENTATION': '#DE33FF',  # Magenta
            'SCIENTIFIC_CHALLENGE': '#FF3333',  # Red
            'METRIC': '#33B5FF',  # Sky blue
            'PROPERTY': '#AAB5FF',  # Light purple
            'AI_FRAMEWORK': '#3339FF',  # Indigo
            'IMAGING_MODALITY': '#FF5588',  # Rose
            'DATA_TYPE': '#58FF88',  # Mint green
            'DISEASE': '#FF0000',  # Bright red
            'SPECIES': '#00FF00',  # Bright green
            'TIME_PERIOD': '#0000FF',  # Bright blue
            'LOCATION': '#FFFF00',  # Yellow
            'TAXONOMY': '#FF00FF',  # Magenta
            'EVOLUTIONARY_PROCESS': '#00FFFF',  # Cyan
            'CONCEPT': '#888888',  # Gray
            'PERSON': '#FFC0CB',  # Pink
            'GPE': '#D3D3D3',  # Light gray
            'PRODUCT': '#F5DEB3',  # Wheat
            'WORK_OF_ART': '#F0E68C',  # Khaki
            # Default for other types
            'DEFAULT': '#AAAAAA'  # Light gray
        }
        
        # Statistics
        self.stats = {
            "num_nodes": 0,
            "num_edges": 0,
            "entity_types": {},
            "relationship_types": {}
        }
    
    def build_graph_from_directories(self):
        """
        Build the knowledge graph from entity and relationship files in the specified directories.
        """
        if not self.entities_dir or not self.relationships_dir:
            print("Both entities directory and relationships directory must be specified.")
            return
        
        # Get all entity and relationship JSON files
        entity_files = list(self.entities_dir.glob('*_entities.json'))
        relationship_files = list(self.relationships_dir.glob('*_relationships.json'))
        
        if not entity_files:
            print(f"No entity files found in {self.entities_dir}")
            return
        
        if not relationship_files:
            print(f"No relationship files found in {self.relationships_dir}")
            return
        
        print(f"Found {len(entity_files)} entity files and {len(relationship_files)} relationship files")
        
        # Load all entities
        all_entities = {}
        
        print("Loading entities...")
        for entity_file in tqdm(entity_files):
            with open(entity_file, 'r', encoding='utf-8') as file:
                entities = json.load(file)
            
            document_id = entity_file.stem.replace('_entities', '')
            
            for entity in entities:
                entity_text = entity['text']
                entity_type = entity['type']
                
                # Normalize entity text to lowercase for consistency
                entity_text = entity_text.lower()
                
                # Add to global entity dictionary if not already present
                if entity_text not in all_entities:
                    all_entities[entity_text] = {
                        'type': entity_type,
                        'documents': [document_id],
                        'count': 1
                    }
                else:
                    # Update entity information
                    if document_id not in all_entities[entity_text]['documents']:
                        all_entities[entity_text]['documents'].append(document_id)
                    all_entities[entity_text]['count'] += 1
        
        # Add entities to the graph
        print("Adding entities to graph...")
        for entity_text, entity_info in all_entities.items():
            entity_type = entity_info['type']
            count = entity_info['count']
            documents = entity_info['documents']
            
            # Add node to the graph
            self.graph.add_node(
                entity_text,
                type=entity_type,
                count=count,
                documents=documents,
                label=entity_text  # Label for visualization
            )
            
            # Update statistics
            if entity_type not in self.stats["entity_types"]:
                self.stats["entity_types"][entity_type] = 0
            self.stats["entity_types"][entity_type] += 1
        
        # Load all relationships
        print("Loading relationships...")
        for relationship_file in tqdm(relationship_files):
            with open(relationship_file, 'r', encoding='utf-8') as file:
                relationships = json.load(file)
            
            document_id = relationship_file.stem.replace('_relationships', '')
            
            for relationship in relationships:
                subject_text = relationship['subject']['text'].lower()
                rel_type = relationship['relationship']
                object_text = relationship['object']['text'].lower()
                
                # Check if both subject and object are in the graph
                if subject_text in all_entities and object_text in all_entities:
                    # Add edge to the graph if not already present
                    if not self.graph.has_edge(subject_text, object_text):
                        self.graph.add_edge(
                            subject_text,
                            object_text,
                            type=rel_type,
                            documents=[document_id],
                            weight=1,
                            label=rel_type  # Label for visualization
                        )
                    else:
                        # Update edge information
                        edge_data = self.graph.get_edge_data(subject_text, object_text)
                        if document_id not in edge_data['documents']:
                            edge_data['documents'].append(document_id)
                        edge_data['weight'] += 1
                    
                    # Update statistics
                    if rel_type not in self.stats["relationship_types"]:
                        self.stats["relationship_types"][rel_type] = 0
                    self.stats["relationship_types"][rel_type] += 1
        
        # Update overall statistics
        self.stats["num_nodes"] = self.graph.number_of_nodes()
        self.stats["num_edges"] = self.graph.number_of_edges()
        
        print(f"Built knowledge graph with {self.stats['num_nodes']} nodes and {self.stats['num_edges']} edges")
        
        # Save statistics
        stats_path = self.output_dir / "graph_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as file:
            json.dump(self.stats, file, indent=2)
    
    def save_graph_data_as_csv(self):
        """
        Save the knowledge graph data as CSV files for nodes and edges.
        """
        # Create node DataFrame
        nodes_data = []
        for node, data in self.graph.nodes(data=True):
            nodes_data.append({
                'id': node,
                'label': node,
                'type': data.get('type', 'UNKNOWN'),
                'count': data.get('count', 0),
                'num_documents': len(data.get('documents', []))
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_path = self.output_dir / "nodes.csv"
        nodes_df.to_csv(nodes_path, index=False)
        
        # Create edge DataFrame
        edges_data = []
        for source, target, data in self.graph.edges(data=True):
            edges_data.append({
                'source': source,
                'target': target,
                'type': data.get('type', 'UNKNOWN'),
                'weight': data.get('weight', 1),
                'num_documents': len(data.get('documents', []))
            })
        
        edges_df = pd.DataFrame(edges_data)
        edges_path = self.output_dir / "edges.csv"
        edges_df.to_csv(edges_path, index=False)
        
        print(f"Saved graph data as CSV files: {nodes_path} and {edges_path}")
    
    def visualize_graph(self, max_nodes=100, filter_min_count=1):
        """
        Create an interactive visualization of the knowledge graph.
        
        Args:
            max_nodes: Maximum number of nodes to include in the visualization
            filter_min_count: Minimum entity count to include in the visualization
        """
        # Filter nodes by count
        filtered_graph = self.graph.copy()
        nodes_to_remove = []
        
        for node, data in filtered_graph.nodes(data=True):
            if data.get('count', 0) < filter_min_count:
                nodes_to_remove.append(node)
        
        filtered_graph.remove_nodes_from(nodes_to_remove)
        
        # If still too many nodes, select the top N by count
        if filtered_graph.number_of_nodes() > max_nodes:
            print(f"Graph has {filtered_graph.number_of_nodes()} nodes, limiting to {max_nodes}")
            
            # Sort nodes by count
            sorted_nodes = sorted(
                [(node, data.get('count', 0)) for node, data in filtered_graph.nodes(data=True)],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Keep only the top max_nodes
            nodes_to_keep = [node for node, _ in sorted_nodes[:max_nodes]]
            
            # Create a new subgraph with only these nodes
            filtered_graph = filtered_graph.subgraph(nodes_to_keep).copy()
        
        # Create a Pyvis network
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True
        )
        
        # Add nodes to the network
        for node, data in filtered_graph.nodes(data=True):
            entity_type = data.get('type', 'DEFAULT')
            count = data.get('count', 0)
            
            # Get color for entity type
            color = self.entity_colors.get(entity_type, self.entity_colors['DEFAULT'])
            
            # Scale node size based on count
            size = 10 + min(count * 5, 40)  # Cap at 50
            
            # Create label
            title = f"<b>{node}</b><br>Type: {entity_type}<br>Count: {count}"
            
            # Add node
            net.add_node(
                node,
                label=node,
                title=title,
                color=color,
                size=size
            )
        
        # Add edges to the network
        for source, target, data in filtered_graph.edges(data=True):
            rel_type = data.get('type', 'UNKNOWN')
            weight = data.get('weight', 1)
            
            # Scale edge width based on weight
            width = 1 + min(weight, 5)  # Cap at 6
            
            # Create label
            title = f"Relationship: {rel_type}<br>Weight: {weight}"
            
            # Add edge
            net.add_edge(
                source,
                target,
                title=title,
                label=rel_type,
                width=width
            )
        
        # Configure physics
        net.barnes_hut(
            gravity=-5000,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.05
        )
        
        # Generate interactive visualization
        output_path = self.output_dir / "knowledge_graph_visualization.html"
        net.save_graph(str(output_path))
        
        print(f"Saved interactive visualization to {output_path}")
        
        return filtered_graph
    
    def create_subgraph_visualization(self, central_node, max_distance=2, max_nodes=50):
        """
        Create a visualization of a subgraph centered around a specific node.
        
        Args:
            central_node: The central node to focus on
            max_distance: Maximum distance from central node
            max_nodes: Maximum number of nodes to include
        """
        if central_node not in self.graph:
            print(f"Node '{central_node}' not found in the graph")
            return
        
        # Create a subgraph centered on the central node
        subgraph = nx.ego_graph(self.graph, central_node, radius=max_distance, undirected=True)
        
        # If still too many nodes, select the top N by count
        if subgraph.number_of_nodes() > max_nodes:
            print(f"Subgraph has {subgraph.number_of_nodes()} nodes, limiting to {max_nodes}")
            
            # Sort nodes by count, ensuring central node is included
            sorted_nodes = sorted(
                [(node, data.get('count', 0)) for node, data in subgraph.nodes(data=True) if node != central_node],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Keep only the top max_nodes-1 plus the central node
            nodes_to_keep = [central_node] + [node for node, _ in sorted_nodes[:max_nodes-1]]
            
            # Create a new subgraph with only these nodes
            subgraph = subgraph.subgraph(nodes_to_keep).copy()
        
        # Create a Pyvis network
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True
        )
        
        # Add nodes to the network
        for node, data in subgraph.nodes(data=True):
            entity_type = data.get('type', 'DEFAULT')
            count = data.get('count', 0)
            
            # Get color for entity type
            color = self.entity_colors.get(entity_type, self.entity_colors['DEFAULT'])
            
            # Scale node size based on count and whether it's the central node
            size = 20 + min(count * 2, 30)  # Cap at 50
            if node == central_node:
                size = 40  # Make central node bigger
                color = "#FF0000"  # Make central node red
            
            # Create label
            title = f"<b>{node}</b><br>Type: {entity_type}<br>Count: {count}"
            
            # Add node
            net.add_node(
                node,
                label=node,
                title=title,
                color=color,
                size=size
            )
        
        # Add edges to the network
        for source, target, data in subgraph.edges(data=True):
            rel_type = data.get('type', 'UNKNOWN')
            weight = data.get('weight', 1)
            
            # Scale edge width based on weight
            width = 1 + min(weight, 5)  # Cap at 6
            
            # Create label
            title = f"Relationship: {rel_type}<br>Weight: {weight}"
            
            # Add edge
            net.add_edge(
                source,
                target,
                title=title,
                label=rel_type,
                width=width,
                arrows="to"
            )
        
        # Configure physics for better visualization
        net.barnes_hut(
            gravity=-2000,
            central_gravity=0.1,
            spring_length=200,
            spring_strength=0.05
        )
        
        # Generate interactive visualization
        output_path = self.output_dir / f"subgraph_{central_node.replace(' ', '_')}.html"
        net.save_graph(str(output_path))
        
        print(f"Saved subgraph visualization to {output_path}")
        
        return subgraph
    
    def export_to_neo4j(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        """
        Export the knowledge graph to Neo4j.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        print("Exporting knowledge graph to Neo4j...")
        
        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Add nodes
            for node, data in tqdm(self.graph.nodes(data=True), desc="Adding nodes"):
                entity_type = data.get('type', 'Entity')
                count = data.get('count', 0)
                
                # Create node with properties
                session.run(
                    """
                    CREATE (n:Entity {name: $name, type: $type, count: $count})
                    """,
                    name=node, type=entity_type, count=count
                )
            
            # Add relationships
            for source, target, data in tqdm(self.graph.edges(data=True), desc="Adding relationships"):
                rel_type = data.get('type', 'RELATED_TO')
                weight = data.get('weight', 1)
                
                # Create normalized relationship name (no spaces, all caps)
                rel_name = rel_type.replace(' ', '_').upper()
                
                # Create relationship with properties
                session.run(
                    f"""
                    MATCH (a:Entity {{name: $source}})
                    MATCH (b:Entity {{name: $target}})
                    CREATE (a)-[r:{rel_name} {{type: $rel_type, weight: $weight}}]->(b)
                    """,
                    source=source, target=target, rel_type=rel_type, weight=weight
                )
        
        print("Successfully exported knowledge graph to Neo4j")
        driver.close()
    
    def analyze_graph(self):
        """
        Analyze the knowledge graph and return key metrics.
        """
        if not self.graph:
            print("Graph is empty. Build the graph first.")
            return
        
        analysis = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "entity_type_distribution": dict(Counter([data.get('type', 'UNKNOWN') for _, data in self.graph.nodes(data=True)])),
            "relationship_type_distribution": dict(Counter([data.get('type', 'UNKNOWN') for _, _, data in self.graph.edges(data=True)])),
            "centrality": {}
        }
        
        # Calculate degree centrality for the top 10 nodes
        degree_centrality = nx.degree_centrality(self.graph)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        analysis["centrality"]["degree"] = {node: value for node, value in top_nodes}
        
        # Calculate betweenness centrality for the top 10 nodes (if the graph is not too large)
        if self.graph.number_of_nodes() <= 1000:
            try:
                betweenness_centrality = nx.betweenness_centrality(self.graph)
                top_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                analysis["centrality"]["betweenness"] = {node: value for node, value in top_nodes}
            except Exception as e:
                print(f"Betweenness centrality calculation skipped (graph too large), {e}")
        
        # Calculate eigenvector centrality for the top 10 nodes (if the graph is not too large)
        if self.graph.number_of_nodes() <= 1000:
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
                top_nodes = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                analysis["centrality"]["eigenvector"] = {node: value for node, value in top_nodes}
            except Exception as e:
                print(f"Eigenvector centrality calculation skipped (graph too large or did not converge), {e}")
        
        # Save analysis to file
        analysis_path = self.output_dir / "graph_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as file:
            json.dump(analysis, file, indent=2)
        
        print(f"Graph analysis saved to {analysis_path}")
        return analysis
    
    def save_graph(self, file_format='graphml'):
        """
        Save the knowledge graph to a file.
        Args:
            format: File format to save the graph (graphml, gexf, etc.)
        """
        # Create a copy of the graph for export
        export_graph = self.graph.copy()
        
        # Convert lists to strings in the export graph
        for node, data in export_graph.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, list):
                    export_graph.nodes[node][key] = ';'.join(str(item) for item in value)
        
        for u, v, data in export_graph.edges(data=True):
            for key, value in data.items():
                if isinstance(value, list):
                    export_graph.edges[u, v][key] = ';'.join(str(item) for item in value)
        
        if file_format == 'graphml':
            output_path = self.output_dir / "knowledge_graph.graphml"
            nx.write_graphml(export_graph, output_path)
        elif file_format == 'gexf':
            output_path = self.output_dir / "knowledge_graph.gexf"
            nx.write_gexf(export_graph, output_path)
        else:
            output_path = self.output_dir / f"knowledge_graph.{file_format}"
            if hasattr(nx, f"write_{file_format}"):
                getattr(nx, f"write_{file_format}")(export_graph, output_path)
            else:
                print(f"Format {file_format} not supported")
                return
        
        print(f"Saved knowledge graph to {output_path}")

def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description='Build and visualize a knowledge graph from extracted entities and relationships.')
    parser.add_argument('--entities', required=True, help='Directory containing entity JSON files')
    parser.add_argument('--relationships', required=True, help='Directory containing relationship JSON files')
    parser.add_argument('--output', default='knowledge_graph', help='Directory to save the knowledge graph files')
    parser.add_argument('--max_nodes', type=int, default=100, help='Maximum number of nodes in the visualization')
    parser.add_argument('--filter_min_count', type=int, default=1, help='Minimum entity count to include in visualization')
    parser.add_argument('--central_node', help='Central node for subgraph visualization')
    parser.add_argument('--neo4j', action='store_true', help='Export to Neo4j database')
    parser.add_argument('--neo4j_uri', default='bolt://localhost:7687', help='Neo4j connection URI')
    parser.add_argument('--neo4j_user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j_password', default='password', help='Neo4j password')
    
    args = parser.parse_args()
    
    # Create graph builder
    builder = KnowledgeGraphBuilder(args.entities, args.relationships, args.output)
    
    # Build and save the graph
    builder.build_graph_from_directories()
    builder.save_graph(file_format='graphml')
    builder.save_graph_data_as_csv()
    
    # Analyze the graph
    builder.analyze_graph()
    
    # Create visualization
    builder.visualize_graph(max_nodes=args.max_nodes, filter_min_count=args.filter_min_count)
    
    # Create subgraph visualization if central node is specified
    if args.central_node:
        builder.create_subgraph_visualization(args.central_node)
    
    # Export to Neo4j if requested
    if args.neo4j:
        builder.export_to_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

if __name__ == "__main__":
    main()
