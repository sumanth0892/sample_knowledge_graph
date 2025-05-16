# entity_extractor.py
import os
import sys
import re
import spacy
import pandas as pd
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter

class EntityExtractor:
    """
    A class for extracting domain-specific entities from text.
    """
    
    def __init__(self, domain_terms_file=None, input_dir=None, output_dir=None):
        """
        Initialize the entity extractor.
        
        Args:
            domain_terms_file: Path to file containing domain terms (YAML or Markdown)
            input_dir: Directory containing preprocessed text files
            output_dir: Directory to save extracted entities
        """
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir) if output_dir else Path('extracted_entities')
        
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
        
        # Load domain terms
        self.domain_entities = {}
        if domain_terms_file:
            self.load_domain_terms(domain_terms_file)
        
        # Initialize statistics
        self.stats = {
            "files_processed": 0,
            "entities_extracted": 0,
            "entity_types": Counter()
        }
    
    def load_domain_terms(self, file_path):
        """
        Load domain-specific terms from a file (YAML or Markdown).
        
        Args:
            file_path: Path to the file containing domain terms
        """
        print(f"Loading domain terms from {file_path}...")
        
        if file_path.endswith(('.yml', '.yaml')):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            # Extract entities
            self.domain_entities = {
                term.lower(): entity_type 
                for entity_type, terms in data.get('entities', {}).items() 
                for term in terms
            }
        
        elif file_path.endswith(('.md', '.markdown')):
            # Parse Markdown format
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract entities from Markdown sections
            entity_sections = re.findall(r'## .*Domain Terms\s+- Entities:(.*?)(?=##|\Z)', content, re.DOTALL)
            for section in entity_sections:
                for line in section.strip().split('\n'):
                    if line.strip().startswith('-'):
                        match = re.search(r"['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]", line)
                        if match:
                            term, entity_type = match.groups()
                            self.domain_entities[term.lower()] = entity_type
        
        print(f"Loaded {len(self.domain_entities)} domain entities")
    
    def extract_entities(self, text):
        """
        Extract domain-specific entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of (entity_text, entity_type, start_pos, end_pos) tuples
        """
        entities = []
        
        # Process with SpaCy
        doc = self.nlp(text)
        
        # Extract named entities recognized by SpaCy
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'WORK_OF_ART']:
                entities.append((ent.text, ent.label_, ent.start_char, ent.end_char))
        
        # Extract domain-specific entities using regex
        for term, entity_type in self.domain_entities.items():
            term_pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            matches = term_pattern.finditer(text)
            for match in matches:
                entities.append((match.group(), entity_type, match.start(), match.end()))
        
        # Sort entities by position in the text
        entities.sort(key=lambda x: x[2])
        
        # Update statistics
        self.stats["entities_extracted"] += len(entities)
        for _, entity_type, _, _ in entities:
            self.stats["entity_types"][entity_type] += 1
            
        return entities
    
    def process_file(self, file_path, save=True):
        """
        Process a single text file and extract entities.
        
        Args:
            file_path: Path to the text file
            save: Whether to save the extracted entities to a file
            
        Returns:
            List of extracted entities
        """
        file_path = Path(file_path)
        
        # Read the text file
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return []
        
        # Extract entities from the text
        entities = self.extract_entities(text)
        
        # Save the extracted entities if requested
        if save:
            output_path = self.output_dir / f"{file_path.stem}_entities.json"
            
            # Convert to serializable format
            entities_json = [
                {
                    "text": entity_text,
                    "type": entity_type,
                    "start": start_pos,
                    "end": end_pos
                }
                for entity_text, entity_type, start_pos, end_pos in entities
            ]
            
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(entities_json, file, indent=2)
        
        return entities
    
    def process_directory(self):
        """
        Process all text files in the input directory.
        """
        if not self.input_dir:
            print("No input directory specified.")
            return
        
        # Get all text files in the input directory
        text_files = list(self.input_dir.glob('*.txt'))
        
        if not text_files:
            print(f"No text files found in {self.input_dir}")
            return
        
        print(f"Found {len(text_files)} text files in {self.input_dir}")
        
        # Process each text file
        for file_path in tqdm(text_files, desc="Extracting entities"):
            self.process_file(file_path)
            self.stats["files_processed"] += 1
        
        print(f"Processed {self.stats['files_processed']} files.")
        print(f"Extracted {self.stats['entities_extracted']} entities.")
        print("Entity types:")
        for entity_type, count in self.stats["entity_types"].most_common():
            print(f"  {entity_type}: {count}")
        
        # Save statistics
        stats_path = self.output_dir / "extraction_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as file:
            json.dump(self.stats, file, indent=2)
    
    def create_entity_summary(self):
        """
        Create a summary of all extracted entities.
        """
        if not self.output_dir:
            print("No output directory specified.")
            return
        
        # Get all entity JSON files
        json_files = list(self.output_dir.glob('*_entities.json'))
        
        if not json_files:
            print(f"No entity files found in {self.output_dir}")
            return
        
        # Collect all entities
        all_entities = defaultdict(set)
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                entities = json.load(file)
            
            for entity in entities:
                all_entities[entity['type']].add(entity['text'])
        
        # Create a summary DataFrame
        summary_data = []
        
        for entity_type, entity_texts in all_entities.items():
            for entity_text in entity_texts:
                summary_data.append({
                    'entity_text': entity_text,
                    'entity_type': entity_type,
                    'count': sum(
                        1 for json_file in json_files
                        for entity in json.load(open(json_file, 'r', encoding='utf-8'))
                        if entity['text'] == entity_text and entity['type'] == entity_type
                    )
                })
        
        # Create DataFrame and sort by count
        df = pd.DataFrame(summary_data)
        df = df.sort_values(['entity_type', 'count'], ascending=[True, False])
        
        # Save to CSV
        summary_path = self.output_dir / "entity_summary.csv"
        df.to_csv(summary_path, index=False)
        
        print(f"Entity summary saved to {summary_path}")
        
        return df

def main():
    """
    Main function for command-line usage.
    """
    
    # Check command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python entity_extractor.py <domain_terms_file> <input_directory> [output_directory]")
        return
    
    # Get input and output directories from command-line arguments
    domain_terms_file = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Create extractor, process directory, and create summary
    extractor = EntityExtractor(domain_terms_file, input_dir, output_dir)
    extractor.process_directory()
    extractor.create_entity_summary()

if __name__ == "__main__":
    main()