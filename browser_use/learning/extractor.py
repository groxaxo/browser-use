"""
Extraction pipeline for DOM and vision data.

Extracts items from web pages using:
- DOM traversal (using existing browser-use DOM services)
- OCR for text detection
- Vision embeddings for image understanding
"""

import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
	from PIL import Image

	PIL_AVAILABLE = True
except ImportError:
	PIL_AVAILABLE = False

try:
	import pytesseract

	TESSERACT_AVAILABLE = True
except ImportError:
	TESSERACT_AVAILABLE = False

from browser_use.learning.views import BoundingBox, ItemCandidate, ItemType

if TYPE_CHECKING:
	from browser_use.browser.session import BrowserSession

logger = logging.getLogger(__name__)


class DOMExtractor:
	"""Extract items from the DOM."""

	def __init__(self, browser_session: 'BrowserSession'):
		"""
		Initialize DOM extractor.

		Args:
		    browser_session: Active browser session
		"""
		self.browser_session = browser_session
		self.logger = logger

	async def extract_items(self, page_url: str | None = None) -> list[ItemCandidate]:
		"""
		Extract actionable items from the current page DOM.

		Args:
		    page_url: Optional URL override (uses current page if None)

		Returns:
		    List of ItemCandidate objects
		"""
		page = self.browser_session.page
		if not page:
			return []

		current_url = page_url or page.url
		page_title = await page.title()

		items = []

		try:
			# Get all interactive and visible elements
			selectors = [
				'a[href]',  # Links
				'button',  # Buttons
				'input',  # Input fields
				'select',  # Dropdowns
				'[role="button"]',  # ARIA buttons
				'[onclick]',  # Click handlers
				'h1, h2, h3, h4, h5, h6',  # Headings
				'img[src]',  # Images
			]

			for selector in selectors:
				try:
					elements = await page.query_selector_all(selector)

					for element in elements:
						try:
							# Check if visible
							is_visible = await element.is_visible()
							if not is_visible:
								continue

							# Get bounding box
							box_dict = await element.bounding_box()
							if not box_dict:
								continue

							bbox = BoundingBox(x=box_dict['x'], y=box_dict['y'], width=box_dict['width'], height=box_dict['height'])

							# Get text content
							text_content = await element.inner_text()
							text_content = text_content.strip()[:2000] if text_content else None

							# Get attributes
							tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
							attributes = {}

							# Collect useful attributes
							for attr in ['id', 'class', 'href', 'src', 'alt', 'title', 'aria-label', 'placeholder', 'type']:
								try:
									value = await element.get_attribute(attr)
									if value:
										attributes[attr] = value
								except Exception:
									pass

							# Determine item type
							item_type = self._determine_item_type(tag_name, attributes, text_content)

							# Generate selector (try to get a stable one)
							generated_selector = await self._generate_selector(element)

							# Create candidate
							candidate = ItemCandidate(
								page_url=current_url,
								page_title=page_title,
								selector=generated_selector,
								raw_text=text_content or attributes.get('alt') or attributes.get('aria-label'),
								item_type=item_type,
								bbox=bbox,
								attributes=attributes,
							)

							items.append(candidate)

						except Exception as e:
							logger.debug(f'Failed to extract element: {e}')
							continue

				except Exception as e:
					logger.debug(f'Failed to query selector {selector}: {e}')
					continue

		except Exception as e:
			logger.error(f'Failed to extract DOM items: {e}')

		logger.info(f'Extracted {len(items)} DOM items from {current_url}')
		return items

	def _determine_item_type(self, tag_name: str, attributes: dict[str, str], text: str | None) -> ItemType:
		"""Determine the type of an item based on its tag and attributes."""
		if tag_name == 'a':
			return ItemType.LINK
		elif tag_name == 'button' or attributes.get('role') == 'button' or 'onclick' in attributes:
			return ItemType.BUTTON
		elif tag_name in ['input', 'select', 'textarea']:
			return ItemType.FIELD
		elif tag_name == 'img':
			return ItemType.IMAGE
		elif tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
			return ItemType.HEADING
		else:
			return ItemType.OTHER

	async def _generate_selector(self, element: Any) -> str | None:
		"""Generate a stable CSS selector for an element."""
		try:
			# Try to use ID first
			element_id = await element.get_attribute('id')
			if element_id:
				return f'#{element_id}'

			# Try data attributes
			for attr in ['data-testid', 'data-id', 'data-cy']:
				value = await element.get_attribute(attr)
				if value:
					return f'[{attr}="{value}"]'

			# Fallback: use Playwright's selector generation
			selector = await element.evaluate(
				'''el => {
                // Simple selector generation
                let selector = el.tagName.toLowerCase();
                if (el.id) return '#' + el.id;
                if (el.className && typeof el.className === 'string') {
                    const classes = el.className.trim().split(/\\s+/).filter(c => c && !c.match(/^ng-|^js-/));
                    if (classes.length > 0) {
                        selector += '.' + classes.slice(0, 2).join('.');
                    }
                }
                return selector;
            }'''
			)
			return selector
		except Exception as e:
			logger.debug(f'Failed to generate selector: {e}')
			return None


class VisionExtractor:
	"""Extract items using OCR and vision models."""

	def __init__(self, browser_session: 'BrowserSession'):
		"""
		Initialize vision extractor.

		Args:
		    browser_session: Active browser session
		"""
		self.browser_session = browser_session
		self.logger = logger

	async def extract_items(
		self, screenshot_path: str | None = None, page_url: str | None = None
	) -> list[ItemCandidate]:
		"""
		Extract items from page screenshot using OCR.

		Args:
		    screenshot_path: Path to save/load screenshot
		    page_url: Page URL for context

		Returns:
		    List of ItemCandidate objects from vision analysis
		"""
		if not PIL_AVAILABLE:
			logger.warning('PIL not available, vision extraction disabled')
			return []

		if not TESSERACT_AVAILABLE:
			logger.warning('Tesseract not available, OCR disabled')
			return []

		page = self.browser_session.page
		if not page:
			return []

		current_url = page_url or page.url
		page_title = await page.title()

		items = []

		try:
			# Take screenshot
			screenshot_bytes = await page.screenshot(full_page=False)
			image = Image.open(io.BytesIO(screenshot_bytes))

			# Save if path provided
			if screenshot_path:
				image.save(screenshot_path)

			# Run OCR
			ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

			# Extract text boxes
			n_boxes = len(ocr_data['text'])
			for i in range(n_boxes):
				text = ocr_data['text'][i].strip()
				if not text or len(text) < 2:
					continue

				conf = int(ocr_data['conf'][i])
				if conf < 30:  # Low confidence
					continue

				# Get bounding box
				x = ocr_data['left'][i]
				y = ocr_data['top'][i]
				w = ocr_data['width'][i]
				h = ocr_data['height'][i]

				bbox = BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))

				# Create candidate
				candidate = ItemCandidate(
					page_url=current_url,
					page_title=page_title,
					raw_text=text,
					item_type=ItemType.OTHER,
					bbox=bbox,
					confidence_score=conf / 100.0,
					attributes={'source': 'ocr', 'confidence': conf},
				)

				items.append(candidate)

		except Exception as e:
			logger.warning(f'Vision extraction failed: {e}')

		logger.info(f'Extracted {len(items)} vision items from {current_url}')
		return items


class ItemConsolidator:
	"""Consolidate and deduplicate items from multiple extraction sources."""

	def __init__(self):
		"""Initialize consolidator."""
		self.logger = logger

	def consolidate(self, dom_items: list[ItemCandidate], vision_items: list[ItemCandidate]) -> list[ItemCandidate]:
		"""
		Merge and deduplicate items from DOM and vision extraction.

		Args:
		    dom_items: Items extracted from DOM
		    vision_items: Items extracted from vision/OCR

		Returns:
		    Consolidated list of unique items
		"""
		# Start with DOM items (more reliable)
		consolidated = list(dom_items)

		# Match vision items to DOM items by bounding box overlap
		for v_item in vision_items:
			if not v_item.bbox:
				continue

			# Check if this vision item overlaps with any DOM item
			matched = False
			for d_item in consolidated:
				if not d_item.bbox:
					continue

				if self._boxes_overlap(v_item.bbox, d_item.bbox, threshold=0.5):
					# Merge text if vision provides additional info
					if v_item.raw_text and not d_item.raw_text:
						d_item.raw_text = v_item.raw_text
					elif v_item.raw_text and d_item.raw_text:
						# Enhance with OCR text if different
						if v_item.raw_text not in d_item.raw_text:
							d_item.attributes['ocr_text'] = v_item.raw_text
					matched = True
					break

			# If no match, add as new item
			if not matched and v_item.raw_text:
				consolidated.append(v_item)

		# Deduplicate by text and position
		unique_items = self._deduplicate(consolidated)

		# Rank by importance
		ranked_items = self._rank_items(unique_items)

		logger.info(f'Consolidated {len(consolidated)} items into {len(ranked_items)} unique items')
		return ranked_items

	def _boxes_overlap(self, box1: BoundingBox, box2: BoundingBox, threshold: float = 0.5) -> bool:
		"""Check if two bounding boxes overlap significantly."""
		# Calculate intersection
		x_left = max(box1.x, box2.x)
		y_top = max(box1.y, box2.y)
		x_right = min(box1.x + box1.width, box2.x + box2.width)
		y_bottom = min(box1.y + box1.height, box2.y + box2.height)

		if x_right < x_left or y_bottom < y_top:
			return False

		intersection_area = (x_right - x_left) * (y_bottom - y_top)
		box1_area = box1.width * box1.height
		box2_area = box2.width * box2.height

		# IoU (Intersection over Union)
		iou = intersection_area / (box1_area + box2_area - intersection_area)
		return iou > threshold

	def _deduplicate(self, items: list[ItemCandidate]) -> list[ItemCandidate]:
		"""Remove duplicate items based on text and position."""
		unique = []
		seen_keys = set()

		for item in items:
			# Create key from normalized text and approximate position
			text_key = (item.raw_text or '')[:100].lower().strip()
			pos_key = f'{int(item.bbox.x / 10) if item.bbox else 0}_{int(item.bbox.y / 10) if item.bbox else 0}'
			key = f'{text_key}_{pos_key}'

			if key not in seen_keys:
				seen_keys.add(key)
				unique.append(item)

		return unique

	def _rank_items(self, items: list[ItemCandidate]) -> list[ItemCandidate]:
		"""Rank items by importance heuristics."""

		def score_item(item: ItemCandidate) -> float:
			score = 0.0

			# Actionable items are more important
			if item.item_type in [ItemType.BUTTON, ItemType.LINK]:
				score += 2.0
			elif item.item_type == ItemType.FIELD:
				score += 1.5

			# Items with selectors are more reliable
			if item.selector:
				score += 1.0

			# Items with text content are more useful
			if item.raw_text and len(item.raw_text) > 3:
				score += 1.0

			# Longer text might be more meaningful (but cap it)
			if item.raw_text:
				score += min(len(item.raw_text) / 100.0, 0.5)

			# Confidence score
			score += item.confidence_score

			return score

		# Sort by score (descending)
		items.sort(key=score_item, reverse=True)
		return items
