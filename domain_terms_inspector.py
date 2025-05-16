# domain_terms_inspector.py
"""
This script thoroughly inspects the domain_terms.md file to diagnose
why relationship types aren't being properly extracted.
"""

import os
import re
from pathlib import Path

def inspect_domain_terms_file(file_path):
    """
    Detailed inspection of the domain terms file structure.
    
    Args:
        file_path: Path to the domain terms file
    """
    try:
        # Check if file exists
        if not Path(file_path).exists():
            print(f"ERROR: File {file_path} does not exist!")
            return
        
        # Read file contents
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Print basic file info
        file_size = len(content)
        line_count = content.count('\n') + 1
        print(f"File: {file_path}")
        print(f"Size: {file_size} bytes")
        print(f"Lines: {line_count}")
        
        # Look for the Relationship Types section
        rel_section_match = re.search(r'## Relationship Types(.*?)(?=##|\Z)', content, re.DOTALL)
        
        if not rel_section_match:
            print("\nWARNING: Could not find '## Relationship Types' section in the file!")
            # Look for similar section headers
            headers = re.findall(r'##\s+(.+)$', content, re.MULTILINE)
            if headers:
                print("Found these section headers instead:")
                for header in headers:
                    print(f"  - {header}")
            return
        
        # Extract and analyze the Relationship Types section
        rel_section = rel_section_match.group(1).strip()
        rel_lines = rel_section.split('\n')
        
        print(f"\nFound Relationship Types section with {len(rel_lines)} lines")
        
        # Analyze the format of relationship entries
        print("\nAnalyzing relationship entry formats:")
        relationship_entries = [line for line in rel_lines if line.strip().startswith('-')]
        print(f"Found {len(relationship_entries)} potential relationship entries")
        
        if relationship_entries:
            # Show sample entries
            print("\nSample relationship entries:")
            for entry in relationship_entries[:5]:
                print(f"  {entry}")
            
            # Check if entries use expected patterns
            single_quote_pattern = re.compile(r"'([^']+)'")
            double_quote_pattern = re.compile(r'"([^"]+)"')
            
            single_quote_matches = [single_quote_pattern.findall(entry) for entry in relationship_entries]
            double_quote_matches = [double_quote_pattern.findall(entry) for entry in relationship_entries]
            
            print("\nQuote patterns found:")
            print(f"  Single quotes: {sum(1 for m in single_quote_matches if m)} entries")
            print(f"  Double quotes: {sum(1 for m in double_quote_matches if m)} entries")
            
            # Try different extraction methods
            print("\nTrying different extraction patterns:")
            
            # Pattern 1: - 'TYPE', 'description'
            pattern1 = r"-\s+['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]"
            matches1 = re.findall(pattern1, rel_section)
            print(f"  Pattern 'TYPE', 'description': {len(matches1)} matches")
            if matches1:
                print(f"    Example: {matches1[0]}")
            
            # Pattern 2: - 'TYPE'
            pattern2 = r"-\s+['\"]([^'\"]+)['\"]"
            matches2 = re.findall(pattern2, rel_section)
            print(f"  Pattern 'TYPE': {len(matches2)} matches")
            if matches2:
                print(f"    Example: {matches2[0]}")
            
            # Pattern 3: - TYPE
            pattern3 = r"-\s+([A-Za-z_]+)"
            matches3 = re.findall(pattern3, rel_section)
            print(f"  Pattern TYPE (no quotes): {len(matches3)} matches")
            if matches3:
                print(f"    Example: {matches3[0]}")
            
            # Try a more relaxed pattern to catch any entries
            pattern4 = r"-\s+(.*)"
            matches4 = re.findall(pattern4, rel_section)
            print(f"  Generic entry pattern: {len(matches4)} matches")
            if matches4:
                print(f"    First few examples:")
                for example in matches4[:3]:
                    print(f"      {example}")
            
        else:
            print("\nWARNING: No lines starting with '-' found in the Relationship Types section!")
            print("Here's the actual content of the section:")
            print("---")
            print(rel_section[:500] + ("..." if len(rel_section) > 500 else ""))
            print("---")
        
        # Check for potential issues
        line_samples = []
        for i, line in enumerate(rel_lines):
            if i < 5 or i >= len(rel_lines) - 5:
                line_samples.append(f"{i+1}: {line}")
                
        print("\nFirst and last few lines of the section:")
        for sample in line_samples:
            print(f"  {sample}")
        
        # Check for specific encoding or formatting issues
        special_chars = set(re.findall(r'[^\w\s\-\'",]', rel_section))
        if special_chars:
            print("\nSpecial characters found in the section:")
            print(f"  {', '.join(special_chars)}")
    
    except Exception as e:
        print(f"ERROR during inspection: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Get the domain terms file path
    domain_terms_file = input("Enter the path to your domain_terms.md file: ")
    
    # If empty, use default
    if not domain_terms_file:
        domain_terms_file = "domain_terms.md"
    
    inspect_domain_terms_file(domain_terms_file)
