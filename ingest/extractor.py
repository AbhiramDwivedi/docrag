"""File parsers for PDF, DOCX, PPTX, XLSX."""
from pathlib import Path
from typing import List, Tuple

Unit = Tuple[str, str]  # logical unit id, text

def extract_text(path: Path) -> List[Unit]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext in {".pptx", ".ppt"}:
        return _extract_pptx(path)
    if ext in {".xlsx", ".xls"}:
        return _extract_xlsx(path)
    if ext == ".txt":
        return _extract_txt(path)
    return []

# ---- private helpers with TODOs ----
def _extract_pdf(path: Path) -> List[Unit]:
    """Extract text from PDF using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF
    try:
        doc = fitz.open(path)
        units = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                units.append((f"page_{page_num + 1}", text))
        doc.close()
        return units
    except Exception as e:
        print(f"Error extracting PDF {path}: {e}")
        return []

def _extract_docx(path: Path) -> List[Unit]:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(str(path))  # Convert Path to string
        text_parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        full_text = "\n".join(text_parts)
        return [("document", full_text)] if full_text.strip() else []
    except Exception as e:
        print(f"Error extracting DOCX {path}: {e}")
        return []

def _extract_pptx(path: Path) -> List[Unit]:
    """Extract text from PPTX using python-pptx."""
    try:
        from pptx import Presentation
        prs = Presentation(str(path))  # Convert Path to string
        units = []
        for slide_num, slide in enumerate(prs.slides):
            text_parts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_parts.append(shape.text)
            slide_text = "\n".join(text_parts)
            if slide_text.strip():
                units.append((f"slide_{slide_num + 1}", slide_text))
        return units
    except Exception as e:
        print(f"Error extracting PPTX {path}: {e}")
        return []

def _extract_xlsx(path: Path) -> List[Unit]:
    """Extract text from XLSX using pandas with size limits and error handling."""
    try:
        import pandas as pd
        
        # Check file size first - skip very large files
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:  # Skip files larger than 50MB
            print(f"Skipping large Excel file {path.name} ({file_size_mb:.1f}MB)")
            return []
        
        # Read Excel file with limited sheets and rows
        excel_file = pd.ExcelFile(path)
        sheet_names = excel_file.sheet_names[:5]  # Limit to first 5 sheets
        
        units = []
        for sheet_name in sheet_names:
            try:
                # Read with strict limits to prevent memory issues and hangs
                df = pd.read_excel(
                    path, 
                    sheet_name=sheet_name, 
                    nrows=500,  # Max 500 rows per sheet
                    engine='openpyxl'  # Explicitly use openpyxl
                )
                
                # Convert dataframe to text representation with limits
                if not df.empty and len(df.columns) <= 50:  # Skip sheets with too many columns
                    text_content = df.to_string(
                        index=False, 
                        max_rows=200,  # Limit display rows
                        max_cols=20    # Limit display columns
                    )
                    
                    # Truncate very long text
                    if text_content.strip():
                        truncated_text = text_content[:5000]  # Max 5KB per sheet
                        units.append((f"sheet_{sheet_name}", truncated_text))
                        
            except Exception as sheet_error:
                print(f"Error processing sheet '{sheet_name}' in {path.name}: {sheet_error}")
                continue
        
        excel_file.close()  # Explicitly close the file
        return units
        
    except Exception as e:
        print(f"Error extracting XLSX {path}: {e}")
        return []

def _extract_txt(path: Path) -> List[Unit]:
    """Extract text from plain text files."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [("document", content)] if content.strip() else []
    except Exception as e:
        print(f"Error extracting TXT {path}: {e}")
        return []
