#!/usr/bin/env python3
"""Enhanced ingestion CLI for Phase 2 with rich metadata capture.

This module extends the basic ingestion process to capture and store
comprehensive metadata for files and emails, enabling advanced queries
like "Find emails from John last week" and "Show newest Excel files".
"""

import argparse
import hashlib
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path so we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.extractor import extract_text, set_all_sheets_mode
from ingest.chunker import chunk_text
from ingest.embed import embed_texts
from ingest.enhanced_vector_store import EnhancedVectorStore
from ingest.extractors.email_extractor import EmailExtractor
from config.config import get_settings
from rich.progress import track
from rich import print as rprint


def extract_file_metadata(path: Path) -> Dict[str, Any]:
    """Extract comprehensive file metadata."""
    try:
        stat = path.stat()
        return {
            'file_path': str(path),
            'file_name': path.name,
            'file_extension': path.suffix.lower(),
            'file_size': stat.st_size,
            'created_time': datetime.fromtimestamp(stat.st_ctime),
            'modified_time': datetime.fromtimestamp(stat.st_mtime),
            'accessed_time': datetime.fromtimestamp(stat.st_atime),
            'file_type': _determine_file_type(path.suffix.lower()),
            'ingestion_time': datetime.now()
        }
    except Exception as e:
        rprint(f"   ⚠️  Could not extract metadata for {path.name}: {e}")
        return {
            'file_path': str(path),
            'file_name': path.name,
            'file_extension': path.suffix.lower(),
            'file_type': _determine_file_type(path.suffix.lower()),
            'ingestion_time': datetime.now()
        }


def extract_email_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Extract email-specific metadata using the email extractor."""
    if path.suffix.lower() not in ['.msg', '.eml']:
        return None
    
    try:
        extractor = EmailExtractor()
        
        # For enhanced metadata, we need to parse the email structure
        if path.suffix.lower() == '.msg':
            return _extract_msg_metadata(path)
        elif path.suffix.lower() == '.eml':
            return _extract_eml_metadata(path)
    except Exception as e:
        rprint(f"   ⚠️  Could not extract email metadata for {path.name}: {e}")
        return None


def _extract_msg_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from .msg file."""
    try:
        import extract_msg
        msg = extract_msg.Message(str(path))
        
        # Extract recipients
        recipients = []
        if hasattr(msg, 'to') and msg.to:
            recipients.extend([addr.strip() for addr in msg.to.split(';') if addr.strip()])
        
        cc_recipients = []
        if hasattr(msg, 'cc') and msg.cc:
            cc_recipients.extend([addr.strip() for addr in msg.cc.split(';') if addr.strip()])
        
        bcc_recipients = []
        if hasattr(msg, 'bcc') and msg.bcc:
            bcc_recipients.extend([addr.strip() for addr in msg.bcc.split(';') if addr.strip()])
        
        # Parse email date
        email_date = None
        if hasattr(msg, 'date') and msg.date:
            try:
                # Try to parse the date string
                import email.utils
                email_date = email.utils.parsedate_to_datetime(msg.date)
            except:
                pass
        
        return {
            'file_path': str(path),
            'message_id': getattr(msg, 'messageId', None) or getattr(msg, 'message_id', None),
            'sender_email': getattr(msg, 'sender', ''),
            'sender_name': getattr(msg, 'senderName', '') or getattr(msg, 'sender_name', ''),
            'recipients': recipients,
            'cc_recipients': cc_recipients,
            'bcc_recipients': bcc_recipients,
            'subject': getattr(msg, 'subject', ''),
            'email_date': email_date,
            'has_attachments': len(getattr(msg, 'attachments', [])) > 0,
            'attachment_count': len(getattr(msg, 'attachments', [])),
            'importance': getattr(msg, 'importance', 'Normal')
        }
        
    except Exception as e:
        rprint(f"   ⚠️  Error extracting MSG metadata: {e}")
        return None


def _extract_eml_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from .eml file."""
    try:
        import email
        from email.utils import parseaddr, parsedate_to_datetime
        
        with open(path, 'rb') as f:
            msg = email.message_from_bytes(f.read())
        
        # Extract sender
        sender_raw = msg.get('From', '')
        sender_name, sender_email = parseaddr(sender_raw)
        
        # Extract recipients
        recipients = []
        to_header = msg.get('To', '')
        if to_header:
            recipients.extend([parseaddr(addr)[1] for addr in to_header.split(',') if addr.strip()])
        
        cc_recipients = []
        cc_header = msg.get('Cc', '')
        if cc_header:
            cc_recipients.extend([parseaddr(addr)[1] for addr in cc_header.split(',') if addr.strip()])
        
        # Parse date
        email_date = None
        date_header = msg.get('Date')
        if date_header:
            try:
                email_date = parsedate_to_datetime(date_header)
            except:
                pass
        
        # Check for attachments
        has_attachments = False
        attachment_count = 0
        for part in msg.walk():
            if part.get_content_disposition() == 'attachment':
                has_attachments = True
                attachment_count += 1
        
        return {
            'file_path': str(path),
            'message_id': msg.get('Message-ID', ''),
            'sender_email': sender_email,
            'sender_name': sender_name,
            'recipients': recipients,
            'cc_recipients': cc_recipients,
            'bcc_recipients': [],  # BCC not typically preserved in EML
            'subject': msg.get('Subject', ''),
            'email_date': email_date,
            'has_attachments': has_attachments,
            'attachment_count': attachment_count,
            'importance': msg.get('Importance', 'Normal')
        }
        
    except Exception as e:
        rprint(f"   ⚠️  Error extracting EML metadata: {e}")
        return None


def _determine_file_type(extension: str) -> str:
    """Determine file type from extension."""
    type_mapping = {
        '.pdf': 'PDF',
        '.docx': 'DOCX', '.doc': 'DOC',
        '.pptx': 'PPTX', '.ppt': 'PPT',
        '.xlsx': 'XLSX', '.xls': 'XLS',
        '.txt': 'TXT',
        '.msg': 'EMAIL', '.eml': 'EMAIL',
        '.html': 'HTML', '.htm': 'HTML',
        '.md': 'MARKDOWN'
    }
    return type_mapping.get(extension.lower(), 'OTHER')


def process_file_enhanced(path: Path, store: EnhancedVectorStore, settings):
    """Process a file with enhanced metadata capture."""
    rprint(f"📄 Processing: {path.name}")
    
    # Extract file metadata
    file_metadata = extract_file_metadata(path)
    
    # Extract email metadata if applicable
    email_metadata = extract_email_metadata(path)
    
    # Extract text content using existing extractors
    file_id = hashlib.sha1(str(path).encode()).hexdigest()[:10]
    units = extract_text(path)
    
    all_chunks, texts, meta = [], [], []
    for unit_id, text in units:
        chunks = chunk_text(text, file_id, unit_id,
                           settings.chunk_size, settings.overlap)
        for ch in chunks:
            all_chunks.append(ch)
            texts.append(ch['text'])
            meta.append((ch['id'], str(path), unit_id, ch['text'], path.stat().st_mtime, 1))
    
    if not texts:
        rprint(f"   ⚠️  No content extracted from {path.name}")
        # Still store metadata even if no content
        file_metadata['chunk_count'] = 0
        store.upsert_with_metadata([], [], [], file_metadata, email_metadata)
        return
    
    # Update chunk count in file metadata
    file_metadata['chunk_count'] = len(all_chunks)
    
    # Embed text content
    vectors = embed_texts(texts, settings.embed_model)
    
    # Store in enhanced vector store with metadata
    store.upsert_with_metadata(
        [c['id'] for c in all_chunks], 
        vectors, 
        meta,
        file_metadata,
        email_metadata
    )
    
    rprint(f"   ✅ Stored {len(all_chunks)} chunks with enhanced metadata")


def main():
    """Enhanced ingestion CLI main function."""
    parser = argparse.ArgumentParser(description="Enhanced ingestion with rich metadata support")
    parser.add_argument('--mode', choices=['full', 'incremental'], default='incremental',
                       help='Full reprocessing or incremental updates')
    parser.add_argument('--file-type', help='Process only specific file types (e.g., xlsx, pdf, docx)')
    parser.add_argument('--target', help='Process specific file by name')
    parser.add_argument('--all-sheets', action='store_true', 
                       help='Process ALL sheets in Excel files (removes 15-sheet limit)')
    parser.add_argument('--migrate', action='store_true',
                       help='Migrate existing database to enhanced schema')
    args = parser.parse_args()
    
    settings = get_settings()
    
    # Set all-sheets mode if requested
    if args.all_sheets:
        set_all_sheets_mode(True)
        rprint("🔄 ALL-SHEETS MODE enabled: Will process all Excel sheets")
    
    # Initialize enhanced vector store
    try:
        from ingest.embed import get_embed_model
        embed_model = get_embed_model()
        embed_dim = len(embed_model.encode("test"))
        
        store = EnhancedVectorStore(
            Path(settings.vector_path), 
            Path(settings.db_path), 
            dim=embed_dim
        )
        rprint(f"✅ Enhanced vector store initialized (dim={embed_dim})")
        
    except Exception as e:
        rprint(f"❌ Failed to initialize enhanced vector store: {e}")
        sys.exit(1)
    
    # Handle migration if requested
    if args.migrate:
        rprint("🔄 Migrating existing database to enhanced schema...")
        store.migrate_from_basic_store()
        rprint("✅ Migration completed")
        return
    
    # Get files to process
    if args.target:
        # Process specific file
        target_path = Path(settings.sync_root) / args.target
        if not target_path.exists():
            rprint(f"❌ Target file not found: {target_path}")
            sys.exit(1)
        files = [target_path]
    elif args.file_type:
        # Filter by file type
        files = list(Path(settings.sync_root).rglob(f'*.{args.file_type}'))
    else:
        # All supported files
        supported_extensions = ['*.pdf', '*.docx', '*.doc', '*.pptx', '*.ppt', 
                               '*.xlsx', '*.xls', '*.txt', '*.msg', '*.eml']
        files = []
        for pattern in supported_extensions:
            files.extend(Path(settings.sync_root).rglob(pattern))
    
    if not files:
        rprint("❌ No files found to process")
        return
    
    rprint(f"🚀 Processing {len(files)} files with enhanced metadata capture...")
    
    # Process files
    processed = 0
    errors = 0
    
    for file_path in track(files, description="Processing files..."):
        try:
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            
            # Skip temporary files
            if file_path.name.startswith('~$'):
                continue
            
            process_file_enhanced(file_path, store, settings)
            processed += 1
            
        except Exception as e:
            rprint(f"   ❌ Error processing {file_path.name}: {e}")
            errors += 1
    
    # Show final statistics
    rprint(f"\n📊 Processing Complete:")
    rprint(f"   ✅ Files processed: {processed}")
    if errors > 0:
        rprint(f"   ❌ Errors: {errors}")
    
    # Show collection statistics
    try:
        stats = store.get_file_statistics()
        rprint(f"\n📈 Collection Statistics:")
        rprint(f"   📄 Total files: {stats['total_files']}")
        rprint(f"   📧 Email files: {stats['total_emails']}")
        rprint(f"   🔢 Total chunks: {stats['total_chunks']}")
        rprint(f"   💾 Total size: {stats['total_size_bytes']:,} bytes")
        
        if stats['file_types']:
            rprint(f"   📋 File types:")
            for file_type in stats['file_types'][:5]:
                rprint(f"      • {file_type['type']}: {file_type['count']} files")
    
    except Exception as e:
        rprint(f"   ⚠️  Could not retrieve statistics: {e}")
    
    rprint("\n🎉 Enhanced ingestion completed!")
    rprint("   You can now use advanced queries like:")
    rprint("   • 'Find emails from john@example.com last week'")
    rprint("   • 'Show me the newest Excel files'")
    rprint("   • 'List PDF files larger than 1MB'")


if __name__ == "__main__":
    main()