# relationship_loader_fix.py
"""
This script demonstrates the correct way to load relationship types
from the domain_terms.md file.
"""

import re
from pathlib import Path

def load_domain_relationships(file_path):
    """
    Correctly load domain-specific relationships from a Markdown file.
    
    Args:
        file_path: Path to the file containing domain terms
        
    Returns:
        Dictionary mapping relationship names to relationship types
    """
    domain_relationships = {}
    
    print(f"Loading domain relationships from {file_path}...")
    
    # Parse Markdown format
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract relationships section from Markdown
    rel_section = re.search(r'## Relationship Types(.*?)(?=##|\Z)', content, re.DOTALL)
    
    if rel_section:
        section_text = rel_section.group(1).strip()
        # Print the entire section for debugging
        print(f"Found Relationship Types section:\n{section_text[:200]}...")
        
        # Process each line in the section
        for line in section_text.split('\n'):
            if line.strip().startswith('-'):
                # Try different regex patterns to match relationship definitions
                
                # Pattern 1: - 'RELATIONSHIP', 'description'
                match1 = re.search(r"['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]", line)
                if match1:
                    rel_type, description = match1.groups()
                    domain_relationships[rel_type.lower()] = rel_type
                    continue
                
                # Pattern 2: - 'RELATIONSHIP'
                match2 = re.search(r"['\"]([^'\"]+)['\"]", line)
                if match2:
                    rel_type = match2.group(1)
                    domain_relationships[rel_type.lower()] = rel_type
    
    print(f"Loaded {len(domain_relationships)} domain relationships")
    
    # Print first few relationships for verification
    if domain_relationships:
        print("First few relationships:")
        for i, (rel_name, rel_type) in enumerate(list(domain_relationships.items())[:5]):
            print(f"  {rel_name} -> {rel_type}")
    
    return domain_relationships

def main():
    # Test the function with your domain terms file
    # Replace with the actual path to your domain_terms.md file
    domain_terms_file = "domain_terms.md"
    
    if not Path(domain_terms_file).exists():
        print(f"Error: File {domain_terms_file} not found.")
        return
    
    relationships = load_domain_relationships(domain_terms_file)
    
    # Print statistics
    print(f"\nTotal relationships loaded: {len(relationships)}")
    
    # Print a sample of the loaded relationships
    print("\nSample of loaded relationships:")
    for rel_name, rel_type in list(relationships.items())[:10]:
        print(f"  {rel_name} -> {rel_type}")

if __name__ == "__main__":
    main()
