"""
Simplified XLSX extractor for Excel spreadsheets.

Features:
- Basic multi-sheet processing
- Simple text conversion
- Error-resistant design
- Configurable sheet limit (default 15, can be disabled)
"""
from pathlib import Path
from typing import List, Any, Optional

from .base import BaseExtractor, Unit

# Global variable to control sheet processing limit
_MAX_SHEETS = 15
_all_sheets_mode = False

def set_all_sheets_mode(enabled: bool = True):
    """Enable or disable processing of all sheets (removes 15-sheet limit)."""
    global _all_sheets_mode
    _all_sheets_mode = enabled

def get_max_sheets() -> Optional[int]:
    """Get the current maximum number of sheets to process."""
    return None if _all_sheets_mode else _MAX_SHEETS


class XLSXExtractor(BaseExtractor):
    """Excel spreadsheet extractor with simplified processing."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".xlsx", ".xls"]
    
    def extract(self, path: Path) -> List[Unit]:
        """Extract text from Excel spreadsheets."""
        try:
            import pandas as pd
            
            print(f"📊 Processing Excel file: {path.name}")
            
            # Get all sheet names
            excel_file = pd.ExcelFile(path)
            all_sheets = excel_file.sheet_names
            max_sheets = get_max_sheets()
            
            if max_sheets and len(all_sheets) > max_sheets:
                print(f"   Found {len(all_sheets)} sheets, limiting to first {max_sheets} (use --all-sheets to process all)")
                sheets_to_process = all_sheets[:max_sheets]
            else:
                print(f"   Found {len(all_sheets)} sheets")
                sheets_to_process = all_sheets
            
            units: List[Unit] = []
            
            for sheet_name in sheets_to_process:
                try:
                    # Read the sheet
                    df = pd.read_excel(path, sheet_name=sheet_name)
                    
                    # Check if sheet is empty
                    if df.empty or df.dropna().empty:
                        print(f"   ⚠️  Sheet '{sheet_name}' is empty")
                        continue
                    
                    # Convert to text
                    sheet_text = self._convert_sheet_to_text(df, str(sheet_name))
                    
                    if sheet_text.strip():
                        units.append((f"sheet_{sheet_name}", sheet_text))
                        print(f"   ✅ Processed sheet '{sheet_name}'")
                    
                except Exception as e:
                    print(f"   ⚠️  Error processing sheet '{sheet_name}': {e}")
                    continue
            
            print(f"   ✅ Extracted {len(units)} sheets total")
            return units
            
        except Exception as e:
            self._log_error(path, e)
            return []
    
    def _convert_sheet_to_text(self, df: Any, sheet_name: str) -> str:
        """Convert dataframe to readable text."""
        try:
            # Get basic info
            rows, cols = df.shape
            
            # Build text representation
            text_parts = [f"Sheet: {sheet_name} ({rows} rows, {cols} columns)"]
            
            # Add column headers if they exist
            if hasattr(df, 'columns') and len(df.columns) > 0:
                headers = [str(col) for col in df.columns]
                text_parts.append("Columns: " + ", ".join(headers))
            
            # Convert data to string representation
            # Limit to first 100 rows to avoid huge text blocks
            sample_df = df.head(100) if len(df) > 100 else df
            
            # Convert to string, handling any data types
            csv_string = sample_df.to_csv(index=False, na_rep='')
            text_parts.append(csv_string)
            
            if len(df) > 100:
                text_parts.append(f"... (showing first 100 of {len(df)} rows)")
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            return f"Sheet: {sheet_name} (Error reading content: {e})"
