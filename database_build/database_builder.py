"""
MALDI Imaging Mass Spectrometry Database Management System

Extract metabolite annotations from MALDI imaging figures using Claude AI
and manage them in a CSV database.

Features: Automated m/z extraction, CSV storage, batch processing, tolerance-based search
Dependencies: anthropic, pandas
"""

import anthropic
import pandas as pd
import base64
import os
from pathlib import Path
from typing import List, Dict, Optional


class MALDIDatabase:
    """
    Database manager for MALDI imaging mass spectrometry annotations.
    
    Handles: image reading via Claude AI, m/z extraction, CSV storage, search
    """
    
    def __init__(self, csv_path: str = "maldi_database.csv", api_key: Optional[str] = None):
        """
        Initialize database. Loads existing CSV or creates new one.
        
        Args:
            csv_path: Path to CSV database file
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        """
        self.csv_path = csv_path
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        
        # Load existing database or create new one
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            print(f"Loaded existing database with {len(self.df)} entries from {csv_path}")
        else:
            self.df = pd.DataFrame(columns=[
                'image_filename', 'mz_value', 'metabolite_name', 
                'tissue_type', 'literature_source', 'notes'
            ])
            print(f"Initialized new database at {csv_path}")
    
    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 encoding for API transmission."""
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")
    
    def extract_annotations(self, image_path: str) -> str:
        """
        Extract MALDI annotations from image using Claude AI.
        Returns Claude's text response with m/z values and metabolite names.
        """
        image_data = self.encode_image(image_path)
        
        # Determine MIME type from file extension
        suffix = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp'
        }
        media_type = media_type_map.get(suffix, 'image/jpeg')
        
        # Send to Claude API
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Please analyze this MALDI imaging mass spectrometry image and extract ALL annotations.

For each annotated peak, provide:
- m/z value (exact number)
- Metabolite/molecule name
- Any tissue type mentioned
- Any other relevant notes

Format your response as a structured list:
m/z: [value] | Metabolite: [name] | Tissue: [type] | Notes: [info]

Be precise with m/z values and metabolite names."""
                        }
                    ],
                }
            ],
        )
        
        return message.content[0].text
    
    def parse_claude_response(self, response: str, image_filename: str, 
                            literature_source: str = "") -> List[Dict]:
        """
        Parse Claude's text response into structured database records.
        Looks for lines with m/z values and extracts annotation data.
        """
        records = []
        lines = response.split('\n')
        
        for line in lines:
            if 'm/z:' in line.lower() or 'm/z' in line.lower():
                try:
                    parts = line.split('|')
                    mz_value = metabolite = tissue = notes = ""
                    
                    for part in parts:
                        part = part.strip()
                        if 'm/z' in part.lower():
                            mz_value = part.split(':')[-1].strip()
                        elif 'metabolite' in part.lower():
                            metabolite = part.split(':')[-1].strip()
                        elif 'tissue' in part.lower():
                            tissue = part.split(':')[-1].strip()
                        elif 'notes' in part.lower():
                            notes = part.split(':')[-1].strip()
                    
                    if mz_value and metabolite:
                        records.append({
                            'image_filename': image_filename,
                            'mz_value': mz_value,
                            'metabolite_name': metabolite,
                            'tissue_type': tissue,
                            'literature_source': literature_source,
                            'notes': notes
                        })
                except Exception as e:
                    print(f"Error parsing line: {line}\nError: {e}")
        
        return records
    
    def add_image(self, image_path: str, literature_source: str = ""):
        """
        Process single MALDI image and add to database.
        Remember to call save() to persist changes.
        """
        print(f"Processing: {image_path}")
        
        response = self.extract_annotations(image_path)
        print(f"\nClaude's response:\n{response}\n")
        
        filename = Path(image_path).name
        records = self.parse_claude_response(response, filename, literature_source)
        
        if records:
            new_df = pd.DataFrame(records)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            print(f"Successfully added {len(records)} records from {filename}")
        else:
            print(f"Warning: No valid records extracted from {filename}")
    
    def batch_process(self, image_folder: str, literature_source: str = ""):
        """Process all images in folder. Continues on errors."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
        
        print(f"Found {len(image_files)} images in {image_folder}")
        
        for image_path in image_files:
            try:
                self.add_image(str(image_path), literature_source)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"\nBatch processing complete. Total entries: {len(self.df)}")
    
    def save(self):
        """Save database to CSV file."""
        self.df.to_csv(self.csv_path, index=False)
        print(f"Database saved to {self.csv_path} ({len(self.df)} total entries)")
    
    def search_by_mz(self, mz_value: str, tolerance: float = 0.5) -> pd.DataFrame:
        """Search by m/z value with tolerance (default ±0.5 Da)."""
        try:
            mz_float = float(mz_value)
            self.df['mz_numeric'] = pd.to_numeric(self.df['mz_value'], errors='coerce')
            results = self.df[
                (self.df['mz_numeric'] >= mz_float - tolerance) & 
                (self.df['mz_numeric'] <= mz_float + tolerance)
            ]
            return results.drop('mz_numeric', axis=1)
        except ValueError:
            return self.df[self.df['mz_value'].str.contains(mz_value, case=False, na=False)]
    
    def search_by_metabolite(self, metabolite_name: str) -> pd.DataFrame:
        """Search by metabolite name (case-insensitive substring match)."""
        return self.df[self.df['metabolite_name'].str.contains(
            metabolite_name, case=False, na=False
        )]
    
    def search_by_literature(self, source: str) -> pd.DataFrame:
        """Search by literature source."""
        return self.df[self.df['literature_source'].str.contains(
            source, case=False, na=False
        )]
    
    def get_statistics(self) -> Dict:
        """Get database summary statistics."""
        stats = {
            'total_entries': len(self.df),
            'unique_metabolites': self.df['metabolite_name'].nunique(),
            'unique_images': self.df['image_filename'].nunique(),
            'tissue_types': self.df['tissue_type'].value_counts().to_dict()
        }
        return stats
    
    def export_metabolite_list(self, output_file: str = "metabolite_list.txt"):
        """Export unique metabolite list to text file."""
        unique_metabolites = self.df['metabolite_name'].unique()
        with open(output_file, 'w') as f:
            for metabolite in sorted(unique_metabolites):
                f.write(f"{metabolite}\n")
        print(f"Exported {len(unique_metabolites)} unique metabolites to {output_file}")
    
    def remove_duplicates(self):
        """Remove duplicate entries based on image, m/z, and metabolite."""
        original_count = len(self.df)
        self.df = self.df.drop_duplicates(
            subset=['image_filename', 'mz_value', 'metabolite_name'],
            keep='first'
        )
        removed_count = original_count - len(self.df)
        print(f"Removed {removed_count} duplicate entries")


if __name__ == "__main__":
    print("MALDI Database Management System")
    print("For detailed usage, see USAGE.md")
