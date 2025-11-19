"""
LM Reader service for cleaning and normalizing item content.

Uses LLMs to:
- Clean and canonicalize text
- Extract keywords
- Suggest item types
- Generate summaries
"""

import json
import logging
from typing import TYPE_CHECKING

from browser_use.learning.views import ItemCandidate, ItemType

if TYPE_CHECKING:
	from browser_use.llm.base import BaseChatModel

logger = logging.getLogger(__name__)


class LMReader:
	"""LM-based content reader and normalizer."""

	CLEANING_PROMPT_TEMPLATE = """You are a text normalizer for UI elements. Your task is to clean and categorize web page elements.

Input element:
- RAW_TEXT: {raw_text}
- ATTRIBUTES: {attributes}
- PAGE_TITLE: {page_title}

Output a JSON object with these fields:
- cleaned_text: A short, canonical label (3-8 words max). Remove marketing fluff, keep core meaning.
- type: One of ["link", "button", "field", "product", "image", "heading", "other"]
- summary: 1-2 sentence summary if the text is longer than 20 words, otherwise leave empty
- keywords: Array of 2-5 important tokens (lowercase, no stopwords)

Example:
Input: "Buy now — 50% off! Limited time only!!!"
Output: {{"cleaned_text": "Buy now button", "type": "button", "summary": "", "keywords": ["buy", "purchase", "offer"]}}

Return ONLY valid JSON, no additional text."""

	def __init__(self, llm: 'BaseChatModel | None' = None):
		"""
		Initialize LM Reader.

		Args:
		    llm: Language model for cleaning. If None, uses simple heuristics.
		"""
		self.llm = llm
		self.logger = logger

	async def clean_item(self, item: ItemCandidate) -> ItemCandidate:
		"""
		Clean and normalize an item's content.

		Args:
		    item: ItemCandidate to clean

		Returns:
		    Updated ItemCandidate with cleaned text and metadata
		"""
		if self.llm:
			return await self._clean_with_llm(item)
		else:
			return self._clean_with_heuristics(item)

	async def _clean_with_llm(self, item: ItemCandidate) -> ItemCandidate:
		"""Clean item using LLM."""
		try:
			# Prepare prompt
			raw_text = item.raw_text or ''
			attributes_str = json.dumps(item.attributes, indent=2) if item.attributes else '{}'
			page_title = item.page_title or 'Unknown'

			prompt = self.CLEANING_PROMPT_TEMPLATE.format(
				raw_text=raw_text[:500], attributes=attributes_str[:300], page_title=page_title[:100]
			)

			# Call LLM
			from browser_use.llm.messages import UserMessage

			messages = [UserMessage(content=prompt)]
			response = await self.llm.get_completion(messages)

			# Parse response
			response_text = str(response)

			# Extract JSON from response (handle markdown code blocks)
			if '```json' in response_text:
				json_start = response_text.find('```json') + 7
				json_end = response_text.find('```', json_start)
				json_text = response_text[json_start:json_end].strip()
			elif '```' in response_text:
				json_start = response_text.find('```') + 3
				json_end = response_text.find('```', json_start)
				json_text = response_text[json_start:json_end].strip()
			else:
				# Try to find JSON object
				json_start = response_text.find('{')
				json_end = response_text.rfind('}') + 1
				if json_start >= 0 and json_end > json_start:
					json_text = response_text[json_start:json_end]
				else:
					json_text = response_text

			result = json.loads(json_text)

			# Update item
			item.cleaned_text = result.get('cleaned_text', item.raw_text)
			item.summary = result.get('summary')
			item.keywords = result.get('keywords', [])

			# Update type if provided and valid
			if 'type' in result:
				try:
					item.item_type = ItemType(result['type'])
				except ValueError:
					pass

			logger.debug(f"Cleaned item: '{item.raw_text[:50]}' -> '{item.cleaned_text}'")

		except json.JSONDecodeError as e:
			logger.warning(f'Failed to parse LLM response as JSON: {e}')
			item = self._clean_with_heuristics(item)
		except Exception as e:
			logger.warning(f'LLM cleaning failed: {e}')
			item = self._clean_with_heuristics(item)

		return item

	def _clean_with_heuristics(self, item: ItemCandidate) -> ItemCandidate:
		"""Clean item using simple heuristics."""
		if not item.raw_text:
			return item

		text = item.raw_text.strip()

		# Remove excessive whitespace
		text = ' '.join(text.split())

		# Remove common marketing phrases
		marketing_phrases = [
			'limited time',
			'act now',
			'hurry',
			'click here',
			'learn more',
			'buy now',
			'shop now',
			'get started',
		]

		# Truncate long text
		if len(text) > 100:
			# Keep first meaningful part
			sentences = text.split('.')
			text = sentences[0][:100] + ('...' if len(text) > 100 else '')

		item.cleaned_text = text

		# Extract simple keywords
		words = text.lower().split()
		stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are'}
		keywords = [w for w in words if len(w) > 3 and w not in stopwords][:5]
		item.keywords = keywords

		return item

	async def batch_clean_items(self, items: list[ItemCandidate]) -> list[ItemCandidate]:
		"""
		Clean multiple items.

		Args:
		    items: List of items to clean

		Returns:
		    List of cleaned items
		"""
		cleaned = []
		for item in items:
			try:
				cleaned_item = await self.clean_item(item)
				cleaned.append(cleaned_item)
			except Exception as e:
				logger.warning(f'Failed to clean item: {e}')
				cleaned.append(item)  # Keep original on failure

		return cleaned
