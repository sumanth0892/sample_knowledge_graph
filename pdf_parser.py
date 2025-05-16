# pdf_parser.py
import os
import sys
from pathlib import Path
import PyPDF2
from tqdm import tqdm

class PDFParser:
    """
    A class for extracting text from PDF files.
    """
    
    def __init__(self, input_dir=None, output_dir=None):
        """
        Initialize the PDF parser.
        
        Args:
            input_dir: Directory containing PDF files to parse
            output_dir: Directory to save extracted text files
        """
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir) if output_dir else Path('extracted_text')
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        text = ""
        
        try:
            # Open the PDF file
            with open(pdf_path, 'rb') as file:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get the number of pages
                num_pages = len(pdf_reader.pages)
                
                # Extract text from each page
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
            return text
        
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def process_file(self, pdf_path, save=True):
        """
        Process a single PDF file and optionally save the extracted text.
        
        Args:
            pdf_path: Path to the PDF file
            save: Whether to save the extracted text to a file
            
        Returns:
            Extracted text as a string
        """
        pdf_path = Path(pdf_path)
        
        print(f"Processing {pdf_path.name}...")
        
        # Extract text from the PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Save the extracted text if requested
        if save:
            output_path = self.output_dir / f"{pdf_path.stem}.txt"
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(text)
            print(f"Saved extracted text to {output_path}")
        
        return text
    
    def process_directory(self):
        """
        Process all PDF files in the input directory.
        """
        if not self.input_dir:
            print("No input directory specified.")
            return
        
        # Get all PDF files in the input directory
        pdf_files = list(self.input_dir.glob('*.pdf'))
        
        if not pdf_files:
            print(f"No PDF files found in {self.input_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files in {self.input_dir}")
        
        # Process each PDF file
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            self.process_file(pdf_path)
        
        print(f"Processed {len(pdf_files)} PDF files. Extracted text saved to {self.output_dir}")
    
def main():
    """
    Main function
    """
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <input_directory> [output_directory]")
        return
    
    # Get input and output directories from command-line arguments
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create parser and process directory
    parser = PDFParser(input_dir, output_dir)
    parser.process_directory()

if __name__ == "__main__":
    main()
