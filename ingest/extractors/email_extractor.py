"""
Email extractor for .msg and .eml files.

Features:
- Thread-level extraction maintaining conversation context
- Complete metadata extraction (sender, recipients, dates, etc.)
- Aggressive signature and quote removal
- Support for .msg (Outlook) and .eml (standard) formats
- All-level thread parsing regardless of depth
"""
import re
import email
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from email.utils import parsedate_to_datetime, parseaddr
from email.message import EmailMessage, Message

try:
    import extract_msg
    from extract_msg import Message as ExtractMsgMessage
except ImportError:
    extract_msg = None
    ExtractMsgMessage = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from .base import BaseExtractor, Unit


class EmailExtractor(BaseExtractor):
    """Email extractor supporting .msg and .eml formats with thread-level processing."""
    
    def __init__(self):
        super().__init__()
        self.signature_patterns = [
            # Common signature delimiters
            r'^--\s*$',
            r'^_{2,}$',
            r'^-{2,}$',
            r'^\s*Sent from my \w+',
            r'^\s*Get Outlook for \w+',
            r'^\s*This email was sent from',
            # Corporate signatures
            r'^\s*Best regards?\s*,?\s*$',
            r'^\s*Kind regards?\s*,?\s*$',
            r'^\s*Thanks?\s*,?\s*$',
            r'^\s*Sincerely\s*,?\s*$',
            # Email disclaimers
            r'^\s*CONFIDENTIAL',
            r'^\s*This e-?mail',
            r'^\s*NOTICE:',
            r'^\s*DISCLAIMER',
        ]
        
        self.quote_patterns = [
            # Reply indicators
            r'^On .+ wrote:',
            r'^From: .+',
            r'^Sent: .+',
            r'^To: .+',
            r'^Subject: .+',
            r'^> .*',  # Quoted lines
            r'^\s*-----Original Message-----',
            r'^\s*________________________________',
            r'^\s*From: \[mailto:',
        ]
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".msg", ".eml"]
    
    def extract(self, path: Path) -> List[Unit]:
        """Extract email thread from .msg or .eml file."""
        try:
            if path.suffix.lower() == ".msg":
                return self._extract_msg(path)
            elif path.suffix.lower() == ".eml":
                return self._extract_eml(path)
            else:
                print(f"   ⚠️  Unsupported email format: {path.suffix}")
                return []
        except Exception as e:
            self._log_error(path, e)
            return []
    
    def _extract_msg(self, path: Path) -> List[Unit]:
        """Extract from Outlook .msg file."""
        if extract_msg is None:
            print(f"   ⚠️  extract-msg not available for {path.name}")
            return []
        
        try:
            print(f"📧 Processing Outlook email: {path.name}")
            
            msg = extract_msg.Message(str(path))
            thread_data = self._parse_msg_thread(msg)
            
            if not thread_data:
                print(f"   ⚠️  No content extracted from {path.name}")
                return []
            
            # Create thread-level unit
            thread_content = self._format_thread_content(thread_data)
            thread_id = self._generate_thread_id(thread_data[0])  # Use first message for ID
            
            print(f"   ✅ Extracted thread with {len(thread_data)} messages")
            return [(thread_id, thread_content)]
            
        except Exception as e:
            print(f"   ❌ Error processing {path.name}: {e}")
            return []
    
    def _extract_eml(self, path: Path) -> List[Unit]:
        """Extract from standard .eml file."""
        try:
            print(f"📧 Processing EML email: {path.name}")
            
            with open(path, 'rb') as f:
                msg = email.message_from_bytes(f.read())
            
            thread_data = self._parse_eml_thread(msg)
            
            if not thread_data:
                print(f"   ⚠️  No content extracted from {path.name}")
                return []
            
            # Create thread-level unit
            thread_content = self._format_thread_content(thread_data)
            thread_id = self._generate_thread_id(thread_data[0])  # Use first message for ID
            
            print(f"   ✅ Extracted thread with {len(thread_data)} messages")
            return [(thread_id, thread_content)]
            
        except Exception as e:
            print(f"   ❌ Error processing {path.name}: {e}")
            return []
    
    def _parse_msg_thread(self, msg: Any) -> List[Dict[str, Any]]:
        """Parse Outlook message and any embedded messages."""
        messages = []
        
        try:
            # Extract main message
            main_msg = self._extract_msg_data(msg)
            if main_msg:
                messages.append(main_msg)
            
            # Look for embedded messages (replies/forwards)
            # This is a simplified approach - full threading would require
            # parsing the message body for embedded email content
            
        except Exception as e:
            print(f"   ⚠️  Error parsing MSG thread: {e}")
        
        return messages
    
    def _parse_eml_thread(self, msg: Union[EmailMessage, Message]) -> List[Dict[str, Any]]:
        """Parse EML message and extract thread information."""
        messages = []
        
        try:
            # Extract main message
            main_msg = self._extract_eml_data(msg)
            if main_msg:
                messages.append(main_msg)
            
            # For EML files, threading info is typically in headers
            # but the actual thread reconstruction would require
            # access to the full mailbox/conversation
            
        except Exception as e:
            print(f"   ⚠️  Error parsing EML thread: {e}")
        
        return messages
    
    def _extract_msg_data(self, msg: Any) -> Optional[Dict[str, Any]]:
        """Extract data from Outlook message object."""
        try:
            # Get sender info
            sender_email = getattr(msg, 'sender', '') or ''
            sender_name = getattr(msg, 'senderName', '') or ''
            
            # Get recipients
            recipients_to = self._parse_recipients(getattr(msg, 'to', ''))
            recipients_cc = self._parse_recipients(getattr(msg, 'cc', ''))
            recipients_bcc = self._parse_recipients(getattr(msg, 'bcc', ''))
            
            # Get dates
            date_sent = getattr(msg, 'date', None)
            if date_sent:
                date_sent = date_sent.isoformat() if hasattr(date_sent, 'isoformat') else str(date_sent)
            
            # Get content
            subject = getattr(msg, 'subject', '') or 'No Subject'
            body = getattr(msg, 'body', '') or ''
            
            # Clean HTML if present
            html_body = getattr(msg, 'htmlBody', '')
            if html_body and not body:
                body = self._html_to_text(html_body)
            elif html_body and len(html_body) > len(body):
                # Use HTML version if it's longer (more content)
                body = self._html_to_text(html_body)
            
            # Clean content
            cleaned_body = self._clean_email_content(body)
            
            # Get attachments
            attachments = []
            try:
                for attachment in getattr(msg, 'attachments', []):
                    if hasattr(attachment, 'longFilename'):
                        attachments.append(attachment.longFilename)
                    elif hasattr(attachment, 'shortFilename'):
                        attachments.append(attachment.shortFilename)
            except:
                pass
            
            return {
                'sender_email': sender_email,
                'sender_name': sender_name,
                'recipients_to': recipients_to,
                'recipients_cc': recipients_cc,
                'recipients_bcc': recipients_bcc,
                'date_sent': date_sent,
                'subject': subject,
                'body': cleaned_body,
                'attachments': attachments,
                'message_type': self._detect_message_type(subject, body),
            }
            
        except Exception as e:
            print(f"   ⚠️  Error extracting MSG data: {e}")
            return None
    
    def _extract_eml_data(self, msg: Union[EmailMessage, Message]) -> Optional[Dict[str, Any]]:
        """Extract data from email.message object."""
        try:
            # Get sender info
            sender_raw = msg.get('From', '')
            sender_name, sender_email = parseaddr(sender_raw)
            
            # Get recipients
            recipients_to = self._parse_recipients(msg.get('To', ''))
            recipients_cc = self._parse_recipients(msg.get('Cc', ''))
            recipients_bcc = self._parse_recipients(msg.get('Bcc', ''))
            
            # Get date
            date_sent = None
            try:
                date_raw = msg.get('Date')
                if date_raw:
                    date_sent = parsedate_to_datetime(date_raw).isoformat()
            except:
                pass
            
            # Get subject
            subject = msg.get('Subject', 'No Subject')
            
            # Get body content
            body = self._extract_email_body(msg)
            cleaned_body = self._clean_email_content(body)
            
            # Get attachments
            attachments = []
            for part in msg.walk():
                if part.get_content_disposition() == 'attachment':
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)
            
            return {
                'sender_email': sender_email,
                'sender_name': sender_name,
                'recipients_to': recipients_to,
                'recipients_cc': recipients_cc,
                'recipients_bcc': recipients_bcc,
                'date_sent': date_sent,
                'subject': subject,
                'body': cleaned_body,
                'attachments': attachments,
                'message_type': self._detect_message_type(subject, body),
            }
            
        except Exception as e:
            print(f"   ⚠️  Error extracting EML data: {e}")
            return None
    
    def _extract_email_body(self, msg) -> str:
        """Extract body content from email message."""
        body = ""
        
        # Handle multipart messages
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = part.get_content_disposition()
                
                # Skip attachments
                if content_disposition == 'attachment':
                    continue
                
                if content_type == 'text/plain':
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        body = part.get_content()
                        if isinstance(body, bytes):
                            body = body.decode(charset, errors='ignore')
                        break  # Prefer plain text
                    except:
                        continue
                
                elif content_type == 'text/html' and not body:
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        html_content = part.get_content()
                        if isinstance(html_content, bytes):
                            html_content = html_content.decode(charset, errors='ignore')
                        body = self._html_to_text(html_content)
                    except:
                        continue
        else:
            # Simple message
            content_type = msg.get_content_type()
            charset = msg.get_content_charset() or 'utf-8'
            
            try:
                content = msg.get_content()
                if isinstance(content, bytes):
                    content = content.decode(charset, errors='ignore')
                
                if content_type == 'text/html':
                    body = self._html_to_text(content)
                else:
                    body = content
            except:
                body = str(msg.get_payload())
        
        return body or ""
    
    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML to clean text."""
        if not BeautifulSoup:
            # Fallback: simple tag removal
            html_content = re.sub(r'<[^>]+>', '', html_content)
            return html_content
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean up
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except:
            # Fallback to simple tag removal
            return re.sub(r'<[^>]+>', '', html_content)
    
    def _clean_email_content(self, content: str) -> str:
        """Aggressively clean email content removing signatures and quotes."""
        if not content:
            return ""
        
        lines = content.split('\n')
        cleaned_lines = []
        in_signature = False
        in_quote = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for signature start
            for pattern in self.signature_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    in_signature = True
                    break
            
            # Check for quote start
            if not in_signature:
                for pattern in self.quote_patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        in_quote = True
                        break
            
            # Skip signature and quoted content
            if in_signature or in_quote:
                continue
            
            # Add line if it has content
            if line_stripped:
                cleaned_lines.append(line)
        
        # Join and clean up extra whitespace
        cleaned_content = '\n'.join(cleaned_lines)
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)  # Max 2 consecutive newlines
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content
    
    def _parse_recipients(self, recipients_str: str) -> List[str]:
        """Parse recipient string into list of email addresses."""
        if not recipients_str:
            return []
        
        recipients = []
        # Split by common delimiters
        for recipient in re.split(r'[;,]', recipients_str):
            recipient = recipient.strip()
            if recipient:
                # Extract email from "Name <email>" format
                name, email_addr = parseaddr(recipient)
                if email_addr:
                    recipients.append(email_addr)
                elif '@' in recipient:
                    recipients.append(recipient)
        
        return recipients
    
    def _detect_message_type(self, subject: str, body: str) -> str:
        """Detect if message is original, reply, or forward."""
        subject_lower = subject.lower()
        body_lower = body.lower()
        
        if subject_lower.startswith(('re:', 'reply:')):
            return 'reply'
        elif subject_lower.startswith(('fwd:', 'fw:', 'forward:')):
            return 'forward'
        elif 'forwarded message' in body_lower or 'original message' in body_lower:
            return 'forward'
        else:
            return 'original'
    
    def _format_thread_content(self, thread_data: List[Dict[str, Any]]) -> str:
        """Format thread data into searchable content."""
        if not thread_data:
            return ""
        
        # Thread header
        first_msg = thread_data[0]
        content_parts = [
            f"Subject: {first_msg.get('subject', 'No Subject')}",
            f"Thread: {len(thread_data)} message{'s' if len(thread_data) != 1 else ''}",
            f"Participants: {self._get_thread_participants(thread_data)}",
            f"Date Range: {self._get_thread_date_range(thread_data)}",
        ]
        
        # Add attachments summary
        all_attachments = []
        for msg in thread_data:
            all_attachments.extend(msg.get('attachments', []))
        if all_attachments:
            content_parts.append(f"Attachments: {', '.join(all_attachments)}")
        
        content_parts.append("")  # Empty line before messages
        
        # Add each message
        for i, msg in enumerate(thread_data, 1):
            msg_header = f"[Message {i}] From: {msg.get('sender_name', '')} <{msg.get('sender_email', '')}>"
            if msg.get('date_sent'):
                msg_header += f" | Date: {msg.get('date_sent', '')}"
            if msg.get('message_type') != 'original':
                msg_header += f" | Type: {msg.get('message_type', '').title()}"
            
            content_parts.append(msg_header)
            
            # Add recipients if multiple people
            recipients = msg.get('recipients_to', [])
            if len(recipients) > 1:
                content_parts.append(f"To: {', '.join(recipients)}")
            
            cc_recipients = msg.get('recipients_cc', [])
            if cc_recipients:
                content_parts.append(f"CC: {', '.join(cc_recipients)}")
            
            # Add message body
            body = msg.get('body', '').strip()
            if body:
                content_parts.append("")  # Empty line before body
                content_parts.append(body)
            
            content_parts.append("")  # Empty line after message
        
        return '\n'.join(content_parts)
    
    def _get_thread_participants(self, thread_data: List[Dict[str, Any]]) -> str:
        """Get unique participants in thread."""
        participants = set()
        for msg in thread_data:
            sender = msg.get('sender_email', '')
            if sender:
                participants.add(sender)
            
            for recipient in msg.get('recipients_to', []):
                participants.add(recipient)
        
        return ', '.join(sorted(participants))
    
    def _get_thread_date_range(self, thread_data: List[Dict[str, Any]]) -> str:
        """Get date range of thread."""
        dates = []
        for msg in thread_data:
            date_sent = msg.get('date_sent')
            if date_sent:
                dates.append(date_sent)
        
        if not dates:
            return "Unknown"
        elif len(dates) == 1:
            return dates[0][:10]  # Just the date part
        else:
            return f"{min(dates)[:10]} to {max(dates)[:10]}"
    
    def _generate_thread_id(self, first_msg: Dict[str, Any]) -> str:
        """Generate thread ID from first message."""
        subject = first_msg.get('subject', 'no_subject')
        sender = first_msg.get('sender_email', 'unknown')
        date = first_msg.get('date_sent', 'unknown')
        
        # Clean subject for ID
        subject_clean = re.sub(r'[^\w\s-]', '', subject.lower())
        subject_clean = re.sub(r'\s+', '_', subject_clean.strip())[:50]
        
        # Get date part
        date_part = date[:10] if date and len(date) >= 10 else 'unknown'
        
        return f"email_thread_{subject_clean}_{date_part}"
