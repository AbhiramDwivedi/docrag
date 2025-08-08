# Documentation

This directory contains comprehensive documentation for the DocQuest project, including architectural guides, implementation details, and processing specifications for various document types.

## Key Documents

### Architecture and Design
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture with component diagrams and data flow patterns
- **[AGENTIC_IMPLEMENTATION_PLAN.md](AGENTIC_IMPLEMENTATION_PLAN.md)** - Multi-agent framework design and plugin architecture

### Implementation Guides
- **[PHASE2_README.md](PHASE2_README.md)** - Phase 2 development roadmap and features
- **[PHASE3_IMPLEMENTATION_GUIDE.md](PHASE3_IMPLEMENTATION_GUIDE.md)** - Advanced features and system enhancements

### Document Processing Specifications
- **[PDF_PROCESSING.md](PDF_PROCESSING.md)** - LangChain integration and AI-powered PDF analysis
- **[EXCEL_PROCESSING.md](EXCEL_PROCESSING.md)** - Enhanced Excel file processing with smart sheet prioritization
- **[EMAIL_PROCESSING.md](EMAIL_PROCESSING.md)** - Email message extraction (.msg and .eml files)
- **[ENHANCED_RAG_IMPLEMENTATION.md](../ENHANCED_RAG_IMPLEMENTATION.md)** - Advanced RAG pipeline features

## Usage Patterns

### For New Contributors
1. Start with [ARCHITECTURE.md](ARCHITECTURE.md) for system overview
2. Review document-specific processing guides for implementation details
3. Consult phase guides for current development priorities

### For Developers
- Reference architecture diagrams when making structural changes
- Update relevant processing documentation when modifying extractors
- Ensure documentation stays current with implementation

## Maintenance

- All architectural changes should update corresponding diagrams in ARCHITECTURE.md
- Document processing changes require updates to format-specific guides
- New features should include documentation updates as part of the implementation

## Links to Implementation

- **Source Code**: [../src/](../src/) - Main implementation directory
- **Tests**: [../tests/](../tests/) - Test suite and test resources
- **Examples**: [../examples/](../examples/) - Demo scripts and usage examples