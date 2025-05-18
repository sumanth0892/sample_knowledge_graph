"""
This is Step 2 in building our knowledge graph
A set of methods to clean and preprocess text data
Step 1 SHOULD HAVE BEEN completed, i.e. the PDF files should've been parsed and 
the results stored.
"""
import re
import sys
import string
from pathlib import Path
from tqdm import tqdm
import nltk

# Download necessary NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    """
    A class to clean and preprocess text data.
    """
    
    def __init__(self, input_dir=None, output_dir=None):
        """
        Initialize the text preprocessor.
        
        Args:
            input_dir: Directory containing text files to preprocess
            output_dir: Directory to save preprocessed text files
        """
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir) if output_dir else Path('preprocessed_text')
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        # Load stopwords
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
    
    def preprocess_text(self, text):
        """
        Preprocess text data.
        Removes URLs and Email addresses.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove citations
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)
        
        # Remove references section (common in academic papers)
        text = re.sub(r'references[\s\S]*$', '', text, flags=re.IGNORECASE)
        
        # Remove page numbers
        text = re.sub(r'\bpage\s+\d+\b', '', text, flags=re.IGNORECASE)
        
        # Convert multiple newlines to a single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Convert multiple spaces to a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (preserving periods for sentence boundaries)
        # We'll keep periods, question marks, and exclamation points for sentence segmentation
        punctuation_to_remove = string.punctuation.replace('.', '').replace('?', '').replace('!', '')
        translator = str.maketrans('', '', punctuation_to_remove)
        text = text.translate(translator)
        
        return text.strip()
    
    def preprocess_file(self, file_path, save=True):
        """
        Process a single text file and optionally save the processed text.
        
        Args:
            file_path: Path to the text file
            save: Whether to save the processed text to a file
            
        Returns:
            Preprocessed text as a string
        """
        file_path = Path(file_path)
        
        # Read the text file
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return ""
        
        # Preprocess the text
        preprocessed_text = self.preprocess_text(text)
        
        # Save the preprocessed text if requested
        if save:
            output_path = self.output_dir / file_path.name
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(preprocessed_text)
        
        return preprocessed_text
    
    def preprocess_directory(self):
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
        for file_path in tqdm(text_files, desc="Preprocessing text files"):
            self.preprocess_file(file_path)
        
        print(f"Preprocessed {len(text_files)} text files. Saved to {self.output_dir}")

def main():
    """
    Main function for command-line usage.
    """
    
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python text_preprocessor.py <input_directory> [output_directory]")
        return
    
    # Get input and output directories from command-line arguments
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create preprocessor and process directory
    preprocessor = TextPreprocessor(input_dir, output_dir)
    preprocessor.preprocess_directory()

if __name__ == "__main__":
    main()