# Excel Processing Features

DocRAG includes advanced Excel processing capabilities designed to handle complex workbooks efficiently and intelligently.

## 🎯 Key Features

### Smart Sheet Prioritization
Automatically prioritizes sheets based on meaningful names:
- **High Priority (Score 8-10)**: Summary, Dashboard, Overview, Data, Analysis, Report
- **Medium Priority (Score 6-7)**: Details, Results, Breakdown
- **Low Priority (Score 1-2)**: Sheet1, Sheet2, temp, test, backup, draft

### Complete Data Extraction
- Extracts all data from sheets (up to 2000 rows per sheet) without sampling
- Preserves data relationships and context
- Converts numerical data to meaningful text representations
- Maintains column headers and row relationships

### Empty Sheet Filtering
- Automatically detects and skips empty or nearly-empty sheets
- Filters out sheets with less than 1% content density
- Improves processing efficiency by focusing on meaningful data

### Processing Limits and Fallback
- **File Size**: Up to 100MB Excel files supported
- **Sheet Limit**: Processes 15 most relevant sheets by default
- **Fallback Strategy**: For workbooks with 15+ sheets, prioritizes by meaningful names
- **All-Sheets Mode**: Optional `--all-sheets` flag to process every sheet

### Detailed Logging
Provides comprehensive progress information:
- Sheet-by-sheet processing status
- Character counts for extracted content
- Empty sheet notifications
- Processing time and statistics
- Clear error reporting

## 📊 Processing Algorithm

1. **Sheet Discovery**: Loads Excel file and identifies all sheets
2. **Prioritization**: Scores sheets based on name patterns and relevance
3. **Selection**: Chooses top 15 sheets (or all if `--all-sheets` enabled)
4. **Processing**: Extracts data sheet-by-sheet with progress logging
5. **Filtering**: Skips empty/meaningless sheets automatically
6. **Text Conversion**: Converts structured data to searchable text with context

## 💻 Command Line Examples

### Basic Excel Processing
```bash
# Process all Excel files with smart prioritization
python -m ingest.ingest --file-type xlsx

# Process specific Excel file
python -m ingest.ingest --file-type xlsx --target "financial_report.xlsx"
```

### Advanced Options
```bash
# Process ALL sheets in a specific file (removes 15-sheet limit)
python -m ingest.ingest --file-type xlsx --target "large_workbook.xlsx" --all-sheets

# Process all Excel files with complete sheet processing
python -m ingest.ingest --file-type xlsx --all-sheets
```

## 🔧 Technical Details

### Sheet Prioritization Patterns
```python
priority_patterns = {
    'summary|overview|dashboard|main|primary|index': 10,
    'data|content|detail|information|info': 8,
    'report|analysis|results|findings': 7,
    'budget|financial|cost|revenue|sales': 6,
    'schedule|timeline|plan|roadmap': 5,
    'config|setting|parameter|metadata': 4,
    'template|example|sample': 2,
    'sheet\d+|temp|tmp|test|backup': 1
}
```

### Text Conversion Process
1. **Headers**: Preserves column names and structure
2. **Data Relationships**: Maintains row-column relationships in text
3. **Numerical Context**: Converts numbers with meaningful labels
4. **Summary Statistics**: Includes data summaries for large numerical datasets
5. **Sheet Identification**: Clearly marks which sheet data comes from

### Performance Characteristics
- **Memory Efficient**: Processes one sheet at a time
- **Error Resilient**: Continues processing if individual sheets fail
- **Progress Tracking**: Real-time feedback on processing status
- **Graceful Degradation**: Falls back to prioritized sheets for large workbooks

## 🚨 Limitations and Considerations

- **File Size**: 100MB maximum file size limit
- **Sheet Complexity**: Very complex formulas may not be fully preserved in text
- **Binary Data**: Images and charts are not extracted
- **Performance**: Large workbooks (20+ sheets) may take longer to process
- **Memory Usage**: Memory consumption scales with sheet size and complexity

## 🛠 Troubleshooting

### Common Issues
- **"Limiting to 15 most relevant sheets"**: Use `--all-sheets` to process all sheets
- **"Empty, skipped"**: Sheet contains no meaningful data (normal behavior)
- **Processing hangs**: Very large sheets may take time; check file size limits
- **Memory errors**: Reduce file size or process specific sheets only

### Best Practices
- Use descriptive sheet names for better prioritization
- Keep individual sheets under 2000 rows for optimal performance
- Use `--target` flag to process specific large files
- Enable `--all-sheets` only when necessary for complete data extraction
