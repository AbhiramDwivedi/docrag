"""
PDF extractor with LangChain integration and AI image analysis.

Features:
- Multi-layer text extraction (LangChain + PyMuPDF fallback)
- GPT-4 Vision analysis of diagrams and charts
- Intelligent image filtering and deduplication
- Adaptive rate limiting for API calls
"""
from pathlib import Path
from typing import List
import time
import io
import base64
import hashlib
import re

from .base import BaseExtractor, Unit

class PDFExtractor(BaseExtractor):
    """Advanced PDF extractor with AI image analysis."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]
    
    def extract(self, path: Path) -> List[Unit]:
        """Extract text and analyze images from PDF using LangChain with intelligent image processing."""
        try:
            print(f"📄 Processing PDF with enhanced LangChain: {path.name}")
            
            # Extract text content
            text_units = self._extract_text(path)
            
            # Extract and analyze images
            print(f"   🖼️  Starting image analysis...")
            image_units = self._extract_and_analyze_images(path)
            
            # Combine results
            all_units = text_units + image_units
            print(f"   🎯 Total extraction: {len(all_units)} units ({len(text_units)} text + {len(image_units)} images)")
            
            return all_units
            
        except Exception as e:
            self._log_error(path, e)
            return []
    
    def _extract_text(self, path: Path) -> List[Unit]:
        """Extract text using multiple LangChain loaders with fallback."""
        units = []
        
        # Method 1: Try UnstructuredPDFLoader
        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
            loader = UnstructuredPDFLoader(str(path))
            documents = loader.load()
            
            for i, doc in enumerate(documents):
                if doc.page_content.strip():
                    page_num = doc.metadata.get('page_number', i + 1)
                    units.append((f"page_{page_num}", doc.page_content))
            
            print(f"   ✅ LangChain extraction successful: {len(units)} pages")
            return units
            
        except ImportError as import_error:
            print(f"   ⚠️  UnstructuredPDFLoader not available ({import_error})")
        except Exception as langchain_error:
            print(f"   ⚠️  LangChain extraction failed ({langchain_error})")
        
        # Method 2: Try PyMuPDFLoader as alternative
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(str(path))
            documents = loader.load()
            
            for i, doc in enumerate(documents):
                if doc.page_content.strip():
                    page_num = doc.metadata.get('page', i + 1)
                    units.append((f"page_{page_num}", doc.page_content))
            
            print(f"   ✅ PyMuPDFLoader extraction successful: {len(units)} pages")
            return units
            
        except Exception as pymupdf_error:
            print(f"   ⚠️  PyMuPDFLoader failed ({pymupdf_error})")
        
        # Method 3: Fallback to direct PyMuPDF
        print(f"   🔄 Falling back to direct PyMuPDF extraction")
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    units.append((f"page_{page_num + 1}", text))
            doc.close()
            print(f"   ✅ PyMuPDF fallback successful: {len(units)} pages")
            return units
            
        except Exception as fallback_error:
            print(f"   ❌ All text extraction methods failed: {fallback_error}")
            return []
    
    def _extract_and_analyze_images(self, path: Path) -> List[Unit]:
        """Extract images from PDF and analyze them with AI."""
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            from openai import OpenAI
            import yaml
            
            # Load OpenAI API key
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
            if not config_path.exists():
                print("   ⚠️  Config file not found - skipping image analysis")
                return []
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                api_key = config.get('openai_api_key')
                if not api_key:
                    print("   ⚠️  No OpenAI API key found - skipping image analysis")
                    return []
            
            client = OpenAI(api_key=api_key)
            pdf_doc = fitz.open(path)
            image_units = []
            
            # Image filtering counters
            seen_image_hashes = set()
            filter_stats = {
                'duplicates': 0,
                'small': 0,
                'aspect_ratio': 0,
                'simple': 0,
                'tiny_data': 0
            }
            
            api_calls_made = 0
            last_api_call_time = 0
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                image_list = page.get_images()
                
                if not image_list:
                    continue
                    
                print(f"   🖼️  Found {len(image_list)} images on page {page_num + 1}")
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image data
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            image = Image.open(io.BytesIO(img_data))
                            
                            # Apply filters
                            if not self._should_process_image(image, img_data, seen_image_hashes, filter_stats):
                                pix = None
                                continue
                            
                            # Check page API call limit
                            page_api_calls = sum(1 for unit_id, _ in image_units if f"page_{page_num + 1}_" in unit_id)
                            if page_api_calls >= 3:
                                print(f"   🛑 Reached maximum API calls limit (3) for page {page_num + 1}")
                                pix = None
                                continue
                            
                            # Rate limiting
                            current_time = time.time()
                            if current_time - last_api_call_time < 0.5:
                                time.sleep(0.5)
                            
                            # Analyze with AI
                            analysis_result = self._analyze_image_with_ai(client, image, api_calls_made)
                            if analysis_result:
                                api_calls_made += 1
                                last_api_call_time = time.time()
                                
                                img_type, content = analysis_result
                                if img_type != 'decorative':
                                    unit_id = f"page_{page_num + 1}_image_{img_index + 1}_{img_type}"
                                    image_units.append((unit_id, f"[IMAGE ANALYSIS]\n{content}"))
                                    print(f"   ✅ Analyzed image {img_index + 1} (type: {img_type})")
                                else:
                                    print(f"   ⚪ AI identified decorative image, skipping")
                            
                        pix = None  # Cleanup
                        
                    except Exception as img_error:
                        print(f"   ⚠️  Failed to process image {img_index + 1} on page {page_num + 1}: {img_error}")
                        continue
            
            pdf_doc.close()
            
            # Report filtering statistics
            self._report_filtering_stats(filter_stats, api_calls_made, len(image_units))
            
            return image_units
            
        except Exception as e:
            print(f"   ⚠️  Image extraction failed: {e}")
            return []
    
    def _should_process_image(self, image, img_data: bytes, seen_hashes: set, filter_stats: dict) -> bool:
        """Apply multiple filters to determine if image should be processed."""
        
        # Filter 1: Size check
        if image.width < 150 or image.height < 150:
            filter_stats['small'] += 1
            return False
        
        # Filter 2: Aspect ratio check
        aspect_ratio = max(image.width, image.height) / min(image.width, image.height)
        if aspect_ratio > 5:
            filter_stats['aspect_ratio'] += 1
            return False
        
        # Filter 3: Deduplication
        img_hash = hashlib.md5(img_data).hexdigest()
        if img_hash in seen_hashes:
            filter_stats['duplicates'] += 1
            return False
        seen_hashes.add(img_hash)
        
        # Filter 4: Color complexity
        colors = image.getcolors(maxcolors=256)
        if colors and len(colors) < 4:
            filter_stats['simple'] += 1
            return False
        
        # Filter 5: File size check
        if len(img_data) < 2048:
            filter_stats['tiny_data'] += 1
            print(f"   ⚪ Skipping tiny image data ({len(img_data)} bytes)")
            return False
        
        return True
    
    def _analyze_image_with_ai(self, client, image, api_calls_made: int):
        """Analyze image with GPT-4 Vision."""
        try:
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image QUICKLY and determine if it's meaningful content.

SKIP if it's:
- Logo, branding, watermark
- Header/footer/page number
- Decorative border/line
- Simple icon or button
- Navigation element

PROCESS if it's:
- Architecture/system diagram
- Flowchart/process diagram
- Technical schematic
- Data chart/graph
- Sequence diagram
- UML diagram
- Screenshot with substantive content

Response format:
IMAGE_TYPE: [diagram/chart/decorative]
CONTENT: [brief description OR "Skip"]"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200
            )
            
            # Check rate limit headers
            if hasattr(response, 'headers'):
                remaining = response.headers.get('x-ratelimit-remaining-requests')
                if remaining and int(remaining) < 5:
                    print(f"   ⏱️  Low API quota remaining ({remaining}), slowing down...")
                    time.sleep(1.0)
            
            analysis = response.choices[0].message.content
            
            # Parse response
            lines = analysis.split('\n')
            img_type = "unknown"
            content = analysis
            
            for line in lines:
                if line.startswith("IMAGE_TYPE:"):
                    img_type = line.replace("IMAGE_TYPE:", "").strip()
                elif line.startswith("CONTENT:"):
                    content = line.replace("CONTENT:", "").strip()
                    break
            
            if img_type.lower() in ['decorative', 'logo'] or 'skip' in content.lower():
                return ('decorative', content)
            else:
                return (img_type, content)
                
        except Exception as vision_error:
            error_msg = str(vision_error)
            
            # Handle rate limiting
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                wait_match = re.search(r'retry after (\d+)', error_msg.lower())
                if wait_match:
                    wait_time = int(wait_match.group(1))
                    print(f"   ⏱️  Rate limited - waiting {wait_time} seconds as requested")
                    time.sleep(wait_time)
                else:
                    print(f"   ⏱️  Rate limited - waiting 5 seconds")
                    time.sleep(5)
                return None
            
            # Handle quota exceeded
            if "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
                print(f"   💰 OpenAI quota exceeded - stopping image analysis")
                return None
                
            print(f"   ⚠️  Vision analysis failed: {vision_error}")
            return None
    
    def _report_filtering_stats(self, filter_stats: dict, api_calls: int, successful_images: int):
        """Report image filtering statistics."""
        print(f"   📊 Image filtering summary:")
        print(f"      • {filter_stats['small']} too small (< 150x150)")
        print(f"      • {filter_stats['aspect_ratio']} wrong aspect ratio (> 5:1)")
        print(f"      • {filter_stats['duplicates']} duplicates")
        print(f"      • {filter_stats['simple']} too simple (< 4 colors)")
        print(f"      • {api_calls} sent to AI for analysis (max 3 per page)")
        
        if successful_images > 0:
            print(f"   🎯 Successfully analyzed {successful_images} meaningful images")
        else:
            print(f"   📄 No meaningful images found (all filtered out)")
