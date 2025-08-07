# Email Processing in DocQuest

DocQuest includes comprehensive email processing capabilities for both Outlook (.msg) and standard (.eml) email formats.

## 🚀 Features

### Thread-Level Processing
- **Context Preservation**: Emails are processed as complete threads, maintaining conversation context
- **Smart Threading**: All conversation levels are captured regardless of thread depth
- **Participant Tracking**: Complete participant lists across the entire thread

### Content Processing
- **Aggressive Signature Removal**: Automatically strips email signatures, disclaimers, and boilerplate
- **Quote Cleaning**: Removes quoted content from replies to focus on new information
- **HTML Support**: Converts HTML emails to clean text while preserving formatting
- **Rich Metadata**: Extracts sender, recipients, dates, attachments, and message types

### Format Support
- **Outlook Messages (.msg)**: Full support for Microsoft Outlook message files
- **Standard Email (.eml)**: RFC 2822 compliant email files
- **Automatic Detection**: Seamless handling of both formats through the same interface

## 📧 What Gets Extracted

### Thread Information
- **Subject Line**: Original email subject
- **Participants**: All unique email addresses in the conversation
- **Date Range**: First to last message timestamps
- **Message Count**: Total messages in thread

### Message Details
- **Sender Information**: Name and email address
- **Recipients**: To, CC, and BCC lists
- **Timestamps**: Sent dates for each message
- **Message Types**: Original, reply, or forward detection
- **Attachments**: File names of all attachments

### Content Processing
```
Original Email:
Hi team,
Here's the project update...

Best regards,
John Smith
Project Manager
--
This email is confidential...

Processed Content:
Hi team,
Here's the project update...
```

## 🔧 Technical Implementation

### Dependencies
```bash
pip install extract-msg beautifulsoup4
```

### Usage Through DocQuest
Email files are automatically processed when found in your document directories:

```bash
# Process all documents including emails
python cli/ask.py --reindex /path/to/documents

# Process only email files
python ingest/ingest.py --folder /path/to/emails --file-type msg
python ingest/ingest.py --folder /path/to/emails --file-type eml
```

### Integration with Vector Search
Processed emails become searchable through DocQuest's vector search:

```bash
python cli/ask.py "project updates from last month"
python cli/ask.py "emails about budget approval"
python cli/ask.py "correspondence with john.smith@company.com"
```

## 📊 Output Format

Each email thread creates a single searchable unit:

```
Subject: Project Update: Q4 Planning
Thread: 1 message
Participants: alice@example.com, bob@example.com
Date Range: 2024-01-15
Attachments: budget_spreadsheet.xlsx, timeline.pdf

[Message 1] From: Alice Smith <alice@example.com> | Date: 2024-01-15T10:30:00 | Type: Original
To: bob@example.com

Hi Bob,

Here's the quarterly update you requested...
```

## 🎯 Best Practices

### For Optimal Results
1. **Export from Email Clients**: Save important conversations as .msg or .eml files
2. **Organize by Project**: Group related emails in project folders
3. **Include Attachments**: Email extractor captures attachment metadata for context
4. **Regular Updates**: Re-index email folders as new messages are added

### Privacy Considerations
- Email processing respects your local-only approach
- No email content is sent to external services
- Metadata extraction happens entirely on your machine
- Vector embeddings preserve privacy while enabling search

## 🔍 Search Capabilities

### Natural Language Queries
- "Show me emails about the budget meeting"
- "Find correspondence with the legal team"
- "What did Sarah say about the project timeline?"

### Metadata-Based Search
- Search by sender, recipients, or date ranges
- Find emails with specific attachment types
- Locate reply chains and conversation threads

### Content-Based Search
- Search within email bodies across all threads
- Find key decisions and action items
- Locate project updates and status reports

## 🛠️ Configuration

Email processing is automatically enabled when DocQuest detects .msg or .eml files. No additional configuration is required.

### File Organization
```
documents/
├── emails/
│   ├── project_alpha/
│   │   ├── kickoff_meeting.msg
│   │   └── status_updates.eml
│   └── team_communications/
│       ├── weekly_reports.msg
│       └── policy_updates.eml
└── other_docs/
    └── ...
```

## 📈 Performance

- **Processing Speed**: ~10-50 emails per second depending on size
- **Memory Usage**: Efficient processing of large email archives
- **Storage**: Thread-level units reduce vector database size
- **Search Speed**: Fast retrieval through optimized embeddings

The email extractor seamlessly integrates with DocQuest's existing document processing pipeline, enabling comprehensive search across all your communication and documentation.
