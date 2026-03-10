"""
Complete workflow: PDF extraction → Database creation → Search
"""

from database_build.pdf_extractor import MALDIFigureExtractor
from database_build.database_builder import MALDIDatabase

# Step 1: Extract MALDI figures from PDFs
print("=" * 60)
print("STEP 1: Extracting MALDI figures from PDFs")
print("=" * 60)

extractor = MALDIFigureExtractor()

# Single PDF
extracted = extractor.extract_from_pdf(
    pdf_path="paper.pdf",  # Replace with your PDF
    output_folder="extracted_figures",
    confidence_threshold=70.0
)

# Multiple PDFs
# results = extractor.batch_extract(
#     pdf_paths=["paper1.pdf", "paper2.pdf"],
#     output_folder="extracted_figures"
# )

# Step 2: Build database from extracted figures
print("\n" + "=" * 60)
print("STEP 2: Building database")
print("=" * 60)

db = MALDIDatabase("metabolite_database.csv")
db.batch_process(
    image_folder="extracted_figures",
    literature_source="DOI: 10.1038/xxxxx"
)
db.save()

# Step 3: Search and analyze
print("\n" + "=" * 60)
print("STEP 3: Searching database")
print("=" * 60)

# Statistics
stats = db.get_statistics()
print(f"\nDatabase stats:")
print(f"  Total entries: {stats['total_entries']}")
print(f"  Unique metabolites: {stats['unique_metabolites']}")
print(f"  Images processed: {stats['unique_images']}")

# Search by m/z
print("\n\nSearch by m/z 885.5 (±0.5):")
results = db.search_by_mz("885.5", tolerance=0.5)
print(results[['mz_value', 'metabolite_name', 'tissue_type']])

# Search by metabolite
print("\n\nSearch for phosphatidylcholine:")
pc_results = db.search_by_metabolite("PC")
print(pc_results[['mz_value', 'metabolite_name']])

# Export
pc_results.to_csv("pc_results.csv", index=False)
print("\nResults exported to pc_results.csv")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
