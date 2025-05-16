# updated_extract_relationships.py
import re

def extract_relationships_from_file(file_path):
    """
    Extract relationships from domain_terms.md without relying on ## headers.
    Looks for a 'Relationship Types' section by keyword matching.
    
    Args:
        file_path: Path to the domain terms file
        
    Returns:
        List of (relationship_type, description) tuples
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for the Relationship Types section by keyword
    relationship_section = None
    
    # Try to find a line containing "Relationship Types" or similar
    relationship_headers = [
        "Relationship Types",
        "Relationships",
        "Relations",
        "Relation Types"
    ]
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if any(header in line for header in relationship_headers):
            # Found a potential section
            print(f"Found potential relationship section marker: '{line}'")
            relationship_section = i
            break
    
    if relationship_section is None:
        print("Couldn't find a relationship section header. Looking for relationship entries directly...")
        # Try to find lines that look like relationship definitions
        relationships = []
        pattern = r"^-\s+['\"]([^'\"]+)['\"]"
        for line in lines:
            match = re.search(pattern, line.strip())
            if match and "PREDICTS" in line or "PART_OF" in line or "USED_FOR" in line:  # Sample relationship types to look for
                rel_type = match.group(1)
                # Check if there's a description
                desc_match = re.search(r"['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]", line)
                if desc_match:
                    rel_type, description = desc_match.groups()
                    relationships.append((rel_type, description))
                else:
                    relationships.append((rel_type, ""))
        
        if relationships:
            print(f"Found {len(relationships)} relationships by direct pattern matching")
            return relationships
        else:
            print("No relationships found by direct pattern matching")
            return []
    
    # Extract list items from the potential relationship section
    relationships = []
    for line in lines[relationship_section+1:]:
        line = line.strip()
        
        # Stop at the next section or an empty line after relationships
        if not line and relationships:
            break
        
        if line.startswith('-'):
            # Extract quoted text
            quote_match = re.search(r"['\"]([^'\"]+)['\"]", line)
            if quote_match:
                rel_type = quote_match.group(1)
                
                # Check if there's a description
                desc_match = re.search(r"['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]", line)
                if desc_match:
                    rel_type, description = desc_match.groups()
                    relationships.append((rel_type, description))
                else:
                    relationships.append((rel_type, ""))
    
    print(f"Found {len(relationships)} relationships from the section")
    
    # Show a sample
    if relationships:
        print("Sample relationships:")
        for rel_type, desc in relationships[:5]:
            if desc:
                print(f"  - '{rel_type}': '{desc}'")
            else:
                print(f"  - '{rel_type}'")
    
    return relationships

def fix_relationship_loader(rel_extractor_file, domain_terms_file):
    """
    Update the relationship extractor code to work with your domain terms file.
    
    Args:
        rel_extractor_file: Path to relationship_extractor.py file
        domain_terms_file: Path to domain_terms.md file
    """
    # Extract relationships from the file
    relationships = extract_relationships_from_file(domain_terms_file)
    
    if not relationships:
        print("No relationships found to fix the loader!")
        return
    
    # Read the relationship extractor file
    with open(rel_extractor_file, 'r', encoding='utf-8') as f:
        extractor_code = f.read()
    
    # Create an updated load_domain_terms method
    updated_method = """
    def load_domain_terms(self, file_path):
        \"\"\"
        Load domain-specific relationships from a file (YAML or Markdown).
        
        This version doesn't rely on ## headers and works with your specific format.
        
        Args:
            file_path: Path to the file containing domain terms
        \"\"\"
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
            # Parse Markdown format without relying on ## headers
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for lines that match relationship patterns
            for line in content.split('\\n'):
                line = line.strip()
                if line.startswith('-'):
                    # Look for pattern: - 'RELATIONSHIP', 'description'
                    # or pattern: - 'RELATIONSHIP'
                    match = re.search(r"['\"]([^'\"]+)['\"]", line)
                    if match:
                        rel_type = match.group(1)
                        self.domain_relationships[rel_type.lower()] = rel_type
        
        print(f"Loaded {len(self.domain_relationships)} domain relationships")
        
        # Add PREDICTS, PART_OF etc. relationships directly if they weren't found
        default_relationships = [
            'PREDICTS', 'PART_OF', 'IMPLEMENTS', 'USED_FOR', 'ANALYZES', 
            'IMPROVES', 'DEVELOPED_BY', 'RELATED_TO', 'INPUT_TO', 'OUTPUT_OF',
            'EVALUATED_BY', 'TRAINED_ON', 'APPLIED_TO', 'DERIVED_FROM',
            'INTERACTS_WITH', 'EVOLVED_FROM', 'FUNCTIONS_AS', 'LOCATED_IN',
            'OCCURS_IN', 'CONSISTS_OF', 'REGULATES', 'BINDS_TO', 'MEASURES',
            'SPECIALIZES', 'USES', 'ACHIEVES', 'SIMILAR_TO', 'DIFFERENT_FROM'
        ]
        
        for rel in default_relationships:
            if rel.lower() not in self.domain_relationships:
                self.domain_relationships[rel.lower()] = rel
                
        print(f"Total relationships (including defaults): {len(self.domain_relationships)}")
        
        # Update relationship patterns based on loaded relationships
        for rel_type in self.domain_relationships.values():
            pattern = f'(\\\\w+[\\\\w\\\\s]*?)\\\\s+{rel_type.lower()}\\\\s+(\\\\w+[\\\\w\\\\s]*)'
            self.relationship_patterns.append((pattern, rel_type))
    """
    
    # Replace the old method with the new one
    updated_code = re.sub(
        r'def load_domain_terms\(self, file_path\):.*?(?=def |$)',
        updated_method,
        extractor_code,
        flags=re.DOTALL
    )
    
    # Save the updated code
    updated_file = "fixed_relationship_extractor.py"
    with open(updated_file, 'w', encoding='utf-8') as f:
        f.write(updated_code)
    
    print(f"Updated relationship extractor saved to {updated_file}")
    print("This version should work with your domain_terms.md format")

if __name__ == "__main__":
    import sys
    
    # Default file paths
    domain_terms_file = "domain_terms.md"
    rel_extractor_file = "relationship_extractor.py"
    
    # Check command-line arguments
    if len(sys.argv) > 1:
        domain_terms_file = sys.argv[1]
    if len(sys.argv) > 2:
        rel_extractor_file = sys.argv[2]
    
    # First just try to extract relationships
    print(f"Attempting to extract relationships from {domain_terms_file}...")
    relationships = extract_relationships_from_file(domain_terms_file)
    
    if relationships:
        # Only try to fix the extractor if we have the file
        if rel_extractor_file and rel_extractor_file != domain_terms_file:
            fix_relationship_loader(rel_extractor_file, domain_terms_file)
    else:
        print("\nFailed to extract any relationships!")
        print("Please manually check your domain_terms.md file format.")
