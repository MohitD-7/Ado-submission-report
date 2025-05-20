# config.py

# --- File Processing Settings ---
# Define terms to be excluded from processing if found in filename (case-insensitive)
EXCLUDE_FILENAME_TERMS = ['Updated', 'Report', 'Sample']
# Define Excel file extensions to process (case-insensitive)
VALID_EXTENSIONS = ('.xlsx', '.xlsm')

# --- Comment Analysis Settings ---
# Primary name of the column containing comments (case-sensitive based on Excel header)
COMMENTS_COLUMN_PRIMARY = 'Comments'
# Fallback column index (0-based) if primary not found (e.g., 7 for Column H)
# Make sure this index exists in your files if used as a fallback.
COMMENTS_COLUMN_FALLBACK_INDEX = 7

# List of terms to count in the comments (case-insensitive matching)
# The script will try to match these as whole words (where applicable)
# and exclude if negated by "not ", "no ", "non ".
TERMS_TO_COUNT = [
    'Bundle/Large', 'Bundle/Similar', 'Parent/Large', 'Kit/Large', 'Kit/Similar',
    'Large', 'Similar', 'Parent', 'Bundle', 'Kit' # 'Kit' should be here
]

# --- SKU Categorization Settings ---
# Define which counted terms contribute to the 'Large / Bundle' category
# INCLUDING the simple 'Kit' term here.
LARGE_BUNDLE_TERMS_CONFIG = ['Large', 'Bundle', 'Bundle/Large', 'Parent/Large', 'Kit/Large', 'Kit'] 
# Define which counted terms contribute to the 'Similar' category
SIMILAR_TERMS_CONFIG = ['Similar', 'Kit/Similar', 'Bundle/Similar']
# Define which counted terms are considered 'Parent' (and typically Regular/Priority)
PARENT_TERMS_CONFIG = ['Parent']

# --- Output Settings ---
# Base name for the output Excel file
OUTPUT_FILENAME_BASE = 'output_results'