"""
Main learning service that orchestrates the learning loop.

Provides high-level API for:
- Extracting and learning from pages
- Querying stored knowledge
- Using learned knowledge for navigation
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from browser_use.learning.database import DatabaseService
from browser_use.learning.embeddings import EmbeddingService, SimpleEmbeddingService
from browser_use.learning.extractor import DOMExtractor, ItemConsolidator, VisionExtractor
from browser_use.learning.reader import LMReader
from browser_use.learning.retrieval import RetrievalService
from browser_use.learning.tagging_cli import SimplifiedTaggingInterface, TaggingCLI
from browser_use.learning.views import ItemCandidate, QueryResult

if TYPE_CHECKING:
	from browser_use.browser.session import BrowserSession
	from browser_use.llm.base import BaseChatModel

logger = logging.getLogger(__name__)


class LearningService:
	"""Main service for the learning loop."""

	def __init__(
		self,
		browser_session: 'BrowserSession',
		llm: 'BaseChatModel | None' = None,
		db_path: str = 'browser_use_learning.db',
		use_vision: bool = False,
		use_openai_embeddings: bool = False,
	):
		"""
		Initialize learning service.

		Args:
		    browser_session: Active browser session
		    llm: Optional LLM for content cleaning
		    db_path: Path to SQLite database
		    use_vision: Enable vision/OCR extraction
		    use_openai_embeddings: Use OpenAI for embeddings (requires API key)
		"""
		self.browser_session = browser_session
		self.llm = llm
		self.use_vision = use_vision

		# Initialize components
		try:
			self.embedding_service = EmbeddingService(use_openai=use_openai_embeddings)
			embedding_dim = self.embedding_service.get_embedding_dimension()
		except Exception as e:
			logger.warning(f'Failed to initialize embedding service, using simple fallback: {e}')
			self.embedding_service = SimpleEmbeddingService()
			embedding_dim = 64

		self.database = DatabaseService(db_path=db_path, vector_dim=embedding_dim)
		self.retrieval = RetrievalService(database=self.database)

		self.dom_extractor = DOMExtractor(browser_session)
		self.vision_extractor = VisionExtractor(browser_session) if use_vision else None
		self.consolidator = ItemConsolidator()

		self.reader = LMReader(llm=llm)
		self.tagging_cli = TaggingCLI(database=self.database)
		self.simplified_tagging = SimplifiedTaggingInterface(database=self.database)

		self.logger = logger

	async def extract_and_learn(
		self, interactive: bool = True, auto_tag: str | None = None, max_items: int = 50
	) -> list[ItemCandidate]:
		"""
		Extract items from current page and optionally tag them.

		Args:
		    interactive: Use interactive CLI for tagging
		    auto_tag: Automatically tag all items with this tag name
		    max_items: Maximum number of items to extract

		Returns:
		    List of extracted items
		"""
		logger.info('Starting extraction and learning...')

		# Extract from DOM
		dom_items = await self.dom_extractor.extract_items()

		# Extract from vision if enabled
		vision_items = []
		if self.vision_extractor:
			vision_items = await self.vision_extractor.extract_items()

		# Consolidate
		all_items = self.consolidator.consolidate(dom_items, vision_items)

		# Limit items
		all_items = all_items[:max_items]

		logger.info(f'Extracted {len(all_items)} items total')

		if not all_items:
			logger.warning('No items extracted')
			return []

		# Clean with LM Reader
		logger.info('Cleaning items with LM Reader...')
		cleaned_items = await self.reader.batch_clean_items(all_items)

		# Generate embeddings
		logger.info('Generating embeddings...')
		await self._add_embeddings(cleaned_items)

		# Tag items
		if auto_tag:
			# Automatic tagging
			self.simplified_tagging.tag_items_bulk(cleaned_items, [auto_tag])
		elif interactive:
			# Interactive tagging
			self.tagging_cli.interactive_tagging_session(cleaned_items)
		else:
			# No tagging, just return items
			pass

		return cleaned_items

	async def _add_embeddings(self, items: list[ItemCandidate]) -> None:
		"""Add embeddings to items."""
		texts = [item.cleaned_text or item.raw_text or '' for item in items]
		texts = [t for t in texts if t]

		if not texts:
			return

		try:
			# Batch embed all texts
			embeddings = self.embedding_service.embed_text_batch(texts)

			# Assign to items
			text_idx = 0
			for item in items:
				text = item.cleaned_text or item.raw_text
				if text and text_idx < len(embeddings):
					item.text_embedding = embeddings[text_idx]
					text_idx += 1
		except Exception as e:
			logger.error(f'Failed to generate embeddings: {e}')

	def query_by_tag(self, tag_name: str) -> dict:
		"""
		Query for items and selectors by tag.

		Args:
		    tag_name: Name of the tag

		Returns:
		    Dictionary with recommended actions
		"""
		return self.retrieval.recommend_actions_for_tag(tag_name)

	def search_similar(self, query_text: str, top_k: int = 5) -> list[QueryResult]:
		"""
		Search for items similar to query text.

		Args:
		    query_text: Text to search for
		    top_k: Number of results

		Returns:
		    List of QueryResult objects
		"""
		# Generate embedding for query
		query_embedding = self.embedding_service.embed_text(query_text)

		if not query_embedding:
			logger.warning('Failed to generate query embedding')
			return []

		# Search
		return self.retrieval.search_similar_items(query_embedding, top_k=top_k)

	def get_tag_statistics(self, tag_name: str) -> dict:
		"""Get statistics about a tag."""
		return self.retrieval.get_tag_statistics(tag_name)

	def list_all_tags(self) -> list[str]:
		"""List all available tags."""
		tags = self.retrieval.get_all_tags()
		return [tag.name for tag in tags]

	async def learn_from_action(
		self, tag_name: str, selector: str, success: bool, page_url: str | None = None
	) -> None:
		"""
		Learn from a navigation action result.

		Args:
		    tag_name: Tag that was being searched for
		    selector: Selector that was used
		    success: Whether the action succeeded
		    page_url: URL where action was performed
		"""
		# Get or create tag
		tag = self.database.get_or_create_tag(tag_name)

		# Update selector confidence
		self.database.update_selector_confidence(tag.id, selector, success=success)

		if success:
			logger.info(f"Learned: Tag '{tag_name}' → Selector '{selector}' (success)")
		else:
			logger.info(f"Learned: Tag '{tag_name}' → Selector '{selector}' (failed)")

	def close(self) -> None:
		"""Clean up resources."""
		self.database.close()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()


async def browse_and_extract(
	browser_session: 'BrowserSession', url: str, llm: 'BaseChatModel | None' = None, auto_tag: str | None = None
) -> list[ItemCandidate]:
	"""
	Convenience function to browse to a URL and extract items.

	Args:
	    browser_session: Browser session
	    url: URL to visit
	    llm: Optional LLM for cleaning
	    auto_tag: Optional tag to automatically apply

	Returns:
	    List of extracted items
	"""
	# Navigate to URL
	page = browser_session.page
	if page:
		await page.goto(url, timeout=30000)

	# Create learning service
	learning_service = LearningService(browser_session, llm=llm)

	# Extract and learn
	items = await learning_service.extract_and_learn(interactive=False, auto_tag=auto_tag)

	learning_service.close()

	return items
