# relationship_extractor.py
import os
import re
import spacy
import pandas as pd
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter

class RelationshipExtractor:
    """
    A class for extracting relationships between entities in text.
    """
    
    def __init__(self, domain_terms_file=None, entities_dir=None, text_dir=None, output_dir=None):
        """
        Initialize the relationship extractor.
        
        Args:
            domain_terms_file: Path to file containing domain relationships (YAML or Markdown)
            entities_dir: Directory containing extracted entity JSON files
            text_dir: Directory containing preprocessed text files
            output_dir: Directory to save extracted relationships
        """
        self.entities_dir = Path(entities_dir) if entities_dir else None
        self.text_dir = Path(text_dir) if text_dir else None
        self.output_dir = Path(output_dir) if output_dir else Path('extracted_relationships')
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        # Load SpaCy model
        print("Loading SpaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
        
        # Load domain relationships
        self.domain_relationships = {}
        self.relationship_patterns = []
        if domain_terms_file:
            self.load_domain_terms(domain_terms_file)
        
        # Define default relationship patterns if none loaded
        if not self.relationship_patterns:
            self.relationship_patterns = [
                (r'(\w+[\w\s]*?)\s+predicts\s+([\w\s]+)', 'PREDICTS'),
                (r'(\w+[\w\s]*?)\s+is a part of\s+([\w\s]+)', 'PART_OF'),
                (r'(\w+[\w\s]*?)\s+implements\s+([\w\s]+)', 'IMPLEMENTS'),
                (r'(\w+[\w\s]*?)\s+is\s+used\s+for\s+([\w\s]+)', 'USED_FOR'),
                (r'(\w+[\w\s]*?)\s+analyzes\s+([\w\s]+)', 'ANALYZES'),
                (r'(\w+[\w\s]*?)\s+improves\s+([\w\s]+)', 'IMPROVES'),
                (r'(\w+[\w\s]*?)\s+was developed by\s+([\w\s]+)', 'DEVELOPED_BY'),
                (r'(\w+[\w\s]*?)\s+is related to\s+([\w\s]+)', 'RELATED_TO'),
                (r'(\w+[\w\s]*?)\s+is\s+trained\s+on\s+([\w\s]+)', 'TRAINED_ON'),
                (r'(\w+[\w\s]*?)\s+interacts with\s+([\w\s]+)', 'INTERACTS_WITH'),
                (r'(\w+[\w\s]*?)\s+binds to\s+([\w\s]+)', 'BINDS_TO'),
                (r'(\w+[\w\s]*?)\s+measures\s+([\w\s]+)', 'MEASURES'),
                (r'(\w+[\w\s]*?)\s+uses\s+([\w\s]+)', 'USES'),
                (r'(\w+[\w\s]*?)\s+consists of\s+([\w\s]+)', 'CONSISTS_OF')
            ]
        
        # Mapping of relationship verbs to relationship types
        self.verb_to_relationship = {
            'predict': 'PREDICTS',
            'generate': 'PREDICTS',
            'produce': 'PREDICTS',
            'create': 'PREDICTS',
            'train': 'TRAINED_ON',
            'use': 'USES',
            'utilize': 'USES',
            'employ': 'USES',
            'develop': 'DEVELOPED_BY',
            'design': 'DEVELOPED_BY',
            'improve': 'IMPROVES',
            'enhance': 'IMPROVES',
            'contain': 'CONSISTS_OF',
            'include': 'CONSISTS_OF',
            'comprise': 'CONSISTS_OF',
            'consist': 'CONSISTS_OF',
            'analyze': 'ANALYZES',
            'study': 'ANALYZES',
            'examine': 'ANALYZES',
            'investigate': 'ANALYZES',
            'interact': 'INTERACTS_WITH',
            'bind': 'BINDS_TO',
            'measure': 'MEASURES',
            'evaluate': 'MEASURES',
            'assess': 'MEASURES',
            'quantify': 'MEASURES',
            'implement': 'IMPLEMENTS',
            'build': 'IMPLEMENTS',
            'construct': 'IMPLEMENTS',
            'relate': 'RELATED_TO',
            'associate': 'RELATED_TO',
            'connect': 'RELATED_TO'
        }
        
        # Initialize statistics
        self.stats = {
            "files_processed": 0,
            "relationships_extracted": 0,
            "relationship_types": Counter()
        }
    
    def load_domain_terms(self, file_path):
        """
        Load domain-specific relationships from a file (YAML or Markdown).
        
        Args:
            file_path: Path to the file containing domain terms
        """
        print(f"Loading domain terms from {file_path}...")
        
        if file_path.endswith(('.yml', '.yaml')):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            # Extract relationships
            self.domain_relationships = {
                rel.lower(): rel_type 
                for rel_type, rels in data.get('relationships', {}).items() 
                for rel in rels
            }
        
        elif file_path.endswith(('.md', '.markdown')):
            # Parse Markdown format
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract relationships from Markdown sections
            rel_section = re.search(r'## Relationship Types(.*?)(?=##|\Z)', content, re.DOTALL)
            if rel_section:
                section_text = rel_section.group(1).strip()
                for line in section_text.split('\n'):
                    if line.strip().startswith("'"):
                        # Extract relationship type
                        match = re.search(r"'([^']+)'", line)
                        if match:
                            rel_type = match.group(1)
                            self.domain_relationships[rel_type.lower()] = rel_type
        
        print(f"Loaded {len(self.domain_relationships)} domain relationships")
        
        # Generate relationship patterns based on loaded relationships
        for rel_type in self.domain_relationships.values():
            pattern = f'(\\w+[\\w\\s]*?)\\s+{rel_type.lower()}\\s+(\\w+[\\w\\s]*)'
            self.relationship_patterns.append((pattern, rel_type))
    
    def extract_relationships_by_position(self, text, entities):
        """
        Extract relationships between entities based on their positions in text.
        
        Args:
            text: The text containing the entities
            entities: List of entity dictionaries with text, type, start, and end positions
            
        Returns:
            List of (subject_entity, relationship_type, object_entity) tuples
        """
        relationships = []
        
        # Sort entities by their position in the text
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        # Analyze text with SpaCy
        doc = self.nlp(text)
        
        # Check for relationships between nearby entities
        for i, entity1 in enumerate(sorted_entities[:-1]):
            # Look at entities within a reasonable distance
            for j in range(i + 1, min(i + 5, len(sorted_entities))):
                entity2 = sorted_entities[j]
                
                # Skip if entities are too far apart (more than 200 characters)
                if entity2['start'] - entity1['end'] > 200:
                    continue
                
                # Extract the text between the two entities
                between_text = text[entity1['end']:entity2['start']]
                
                # Check for relationship patterns in the text between entities
                for pattern, rel_type in self.relationship_patterns:
                    if re.search(pattern, between_text, re.IGNORECASE):
                        relationships.append((entity1, rel_type, entity2))
                
                # Check for verb-based relationships using SpaCy's dependency parsing
                between_doc = self.nlp(between_text)
                for token in between_doc:
                    if token.pos_ == 'VERB' and token.lemma_ in self.verb_to_relationship:
                        rel_type = self.verb_to_relationship[token.lemma_]
                        relationships.append((entity1, rel_type, entity2))
        
        # For sentences that have exact "subject verb object" structure
        for sent in doc.sents:
            for token in sent:
                # Find root verbs
                if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                    # Look for subject of the verb
                    subject_token = None
                    for child in token.children:
                        if child.dep_ in ['nsubj', 'nsubjpass']:
                            subject_token = child
                            break
                    
                    # Look for object of the verb
                    object_token = None
                    for child in token.children:
                        if child.dep_ in ['dobj', 'pobj', 'attr']:
                            object_token = child
                            break
                    
                    # If we found both subject and object
                    if subject_token and object_token:
                        # Find entities that contain the subject and object
                        subject_entity = None
                        object_entity = None
                        
                        for entity in sorted_entities:
                            # Check if entity contains subject token
                            if (subject_token.idx >= entity['start'] and 
                                subject_token.idx + len(subject_token.text) <= entity['end']):
                                subject_entity = entity
                            
                            # Check if entity contains object token
                            if (object_token.idx >= entity['start'] and 
                                object_token.idx + len(object_token.text) <= entity['end']):
                                object_entity = entity
                        
                        # If we found both entities, add the relationship
                        if subject_entity and object_entity:
                            # Get relationship type from verb mapping or use verb itself
                            rel_type = self.verb_to_relationship.get(token.lemma_, token.lemma_.upper())
                            relationships.append((subject_entity, rel_type, object_entity))
        
        return relationships
    
    def process_file(self, entity_file, save=True):
        """
        Process a single entity file and extract relationships.
        
        Args:
            entity_file: Path to the entity JSON file
            save: Whether to save the extracted relationships to a file
            
        Returns:
            List of extracted relationships
        """
        entity_file = Path(entity_file)
        
        # Read the entity file
        try:
            with open(entity_file, 'r', encoding='utf-8') as file:
                entities = json.load(file)
        except Exception as e:
            print(f"Error reading {entity_file}: {str(e)}")
            return []
        
        # Find corresponding text file
        text_file = self.text_dir / f"{entity_file.stem.replace('_entities', '')}.txt"
        if not text_file.exists():
            print(f"Text file {text_file} not found.")
            return []
        
        # Read the text file
        try:
            with open(text_file, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading {text_file}: {str(e)}")
            return []
        
        # Extract relationships from the text and entities
        relationships = self.extract_relationships_by_position(text, entities)
        
        # Save the extracted relationships if requested
        if save:
            output_path = self.output_dir / f"{entity_file.stem.replace('_entities', '')}_relationships.json"
            
            # Convert to serializable format
            relationships_json = [
                {
                    "subject": {
                        "text": subject['text'],
                        "type": subject['type']
                    },
                    "relationship": rel_type,
                    "object": {
                        "text": obj['text'],
                        "type": obj['type']
                    }
                }
                for subject, rel_type, obj in relationships
            ]
            
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(relationships_json, file, indent=2)
        
        # Update statistics
        self.stats["relationships_extracted"] += len(relationships)
        for _, rel_type, _ in relationships:
            self.stats["relationship_types"][rel_type] += 1
        
        return relationships
    
    def process_directory(self):
        """
        Process all entity files in the entities directory.
        """
        if not self.entities_dir or not self.text_dir:
            print("Both entities directory and text directory must be specified.")
            return
        
        # Get all entity JSON files
        entity_files = list(self.entities_dir.glob('*_entities.json'))
        
        if not entity_files:
            print(f"No entity files found in {self.entities_dir}")
            return
        
        print(f"Found {len(entity_files)} entity files in {self.entities_dir}")
        
        # Process each entity file
        for entity_file in tqdm(entity_files, desc="Extracting relationships"):
            self.process_file(entity_file)
            self.stats["files_processed"] += 1
        
        print(f"Processed {self.stats['files_processed']} files.")
        print(f"Extracted {self.stats['relationships_extracted']} relationships.")
        print("Relationship types:")
        for rel_type, count in self.stats["relationship_types"].most_common():
            print(f"  {rel_type}: {count}")
        
        # Save statistics
        stats_path = self.output_dir / "relationship_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as file:
            json.dump(self.stats, file, indent=2)
    
    def create_relationship_summary(self):
        """
        Create a summary of all extracted relationships.
        """
        if not self.output_dir:
            print("No output directory specified.")
            return
        
        # Get all relationship JSON files
        json_files = list(self.output_dir.glob('*_relationships.json'))
        
        if not json_files:
            print(f"No relationship files found in {self.output_dir}")
            return
        
        # Collect all relationships
        all_relationships = []
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                relationships = json.load(file)
            
            document_id = json_file.stem.replace('_relationships', '')
            
            for rel in relationships:
                all_relationships.append({
                    'document_id': document_id,
                    'subject_text': rel['subject']['text'],
                    'subject_type': rel['subject']['type'],
                    'relationship': rel['relationship'],
                    'object_text': rel['object']['text'],
                    'object_type': rel['object']['type']
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_relationships)
        
        # Save to CSV
        summary_path = self.output_dir / "relationship_summary.csv"
        df.to_csv(summary_path, index=False)
        
        print(f"Relationship summary saved to {summary_path}")
        
        return df

def main():
    """
    Main function for command-line usage.
    """
    import sys
    
    # Check command-line arguments
    if len(sys.argv) < 4:
        print("Usage: python relationship_extractor.py <domain_terms_file> <entities_directory> <text_directory> [output_directory]")
        return
    
    # Get input and output directories from command-line arguments
    domain_terms_file = sys.argv[1]
    entities_dir = sys.argv[2]
    text_dir = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Create extractor, process directory, and create summary
    extractor = RelationshipExtractor(domain_terms_file, entities_dir, text_dir, output_dir)
    extractor.process_directory()
    extractor.create_relationship_summary()

if __name__ == "__main__":
    main()
