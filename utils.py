import os
import shutil
from typing import List

def prepare_sample_documents(docs_dir: str, sample_texts: List[str]):
    """Create sample documents for RAG"""
    os.makedirs(docs_dir, exist_ok=True)
    
    for i, text in enumerate(sample_texts):
        with open(os.path.join(docs_dir, f"doc_{i}.txt"), "w") as f:
            f.write(text)

def clear_output_directory(output_dir: str):
    """Clear the output directory"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)