"""
Retrieval service for querying stored knowledge.

Provides methods to:
- Search by tag
- Search by semantic similarity
- Find best selectors for tags
- Get recommendations for current page
"""

import logging
from typing import TYPE_CHECKING

from browser_use.learning.database import DatabaseService
from browser_use.learning.views import QueryResult, SelectorConfidence, StoredItem, Tag

if TYPE_CHECKING:
	pass

logger = logging.getLogger(__name__)


class RetrievalService:
	"""Service for retrieving stored knowledge."""

	def __init__(self, database: DatabaseService):
		"""
		Initialize retrieval service.

		Args:
		    database: DatabaseService instance
		"""
		self.database = database
		self.logger = logger

	def get_items_by_tag(self, tag_name: str) -> list[StoredItem]:
		"""
		Get all items associated with a tag.

		Args:
		    tag_name: Name of the tag

		Returns:
		    List of StoredItem objects
		"""
		return self.database.get_items_by_tag(tag_name)

	def search_similar_items(
		self, query_embedding: list[float], top_k: int = 5, use_text: bool = True
	) -> list[QueryResult]:
		"""
		Search for similar items using vector similarity.

		Args:
		    query_embedding: Query embedding vector
		    top_k: Number of results to return
		    use_text: Use text embeddings (True) or image embeddings (False)

		Returns:
		    List of QueryResult objects sorted by similarity
		"""
		return self.database.search_by_embedding(query_embedding, top_k=top_k, use_text=use_text)

	def get_best_selectors(self, tag_name: str, limit: int = 5) -> list[SelectorConfidence]:
		"""
		Get the most reliable selectors for a tag.

		Args:
		    tag_name: Name of the tag
		    limit: Maximum number of selectors to return

		Returns:
		    List of SelectorConfidence objects sorted by confidence
		"""
		# First get the tag
		cursor = self.database.conn.cursor()
		cursor.execute('SELECT id FROM tags WHERE name = ?', (tag_name,))
		row = cursor.fetchone()

		if not row:
			logger.warning(f"Tag '{tag_name}' not found")
			return []

		tag_id = row['id']
		return self.database.get_best_selectors_for_tag(tag_id, limit=limit)

	def recommend_actions_for_tag(self, tag_name: str, current_page_items: list[str] | None = None) -> dict:
		"""
		Recommend actions for a tag on the current page.

		Args:
		    tag_name: Name of the tag to search for
		    current_page_items: Optional list of selectors present on current page

		Returns:
		    Dictionary with recommended selectors and alternatives
		"""
		result = {'tag': tag_name, 'selectors': [], 'alternatives': []}

		# Get best known selectors for this tag
		best_selectors = self.get_best_selectors(tag_name, limit=5)

		if not best_selectors:
			logger.info(f"No selectors found for tag '{tag_name}'")
			return result

		# Filter by current page if provided
		if current_page_items:
			for sel_conf in best_selectors:
				if sel_conf.selector in current_page_items:
					result['selectors'].append(
						{
							'selector': sel_conf.selector,
							'confidence': sel_conf.confidence,
							'usage_count': sel_conf.usage_count,
							'on_page': True,
						}
					)
		else:
			# Return all best selectors
			for sel_conf in best_selectors:
				result['selectors'].append(
					{
						'selector': sel_conf.selector,
						'confidence': sel_conf.confidence,
						'usage_count': sel_conf.usage_count,
						'on_page': None,  # Unknown
					}
				)

		# Get example items for context
		items = self.get_items_by_tag(tag_name)
		if items:
			# Add a few examples as alternatives
			for item in items[:3]:
				if item.selector:
					result['alternatives'].append(
						{
							'selector': item.selector,
							'text': item.cleaned_text or item.raw_text,
							'page_url': item.page_url,
						}
					)

		return result

	def find_tag_for_text(self, text: str, threshold: float = 0.7) -> list[tuple[Tag, float]]:
		"""
		Find tags that might match given text.

		Args:
		    text: Text to match
		    threshold: Minimum similarity threshold

		Returns:
		    List of (Tag, score) tuples
		"""
		# Simple keyword matching for now (can be enhanced with embeddings)
		cursor = self.database.conn.cursor()
		cursor.execute('SELECT DISTINCT t.* FROM tags t')
		all_tags = cursor.fetchall()

		matches = []
		text_lower = text.lower()

		for tag_row in all_tags:
			tag_name = tag_row['name'].lower()

			# Simple substring matching
			if tag_name in text_lower or text_lower in tag_name:
				# Compute simple score
				score = max(len(tag_name) / len(text_lower), len(text_lower) / len(tag_name))
				score = min(score, 1.0)

				if score >= threshold:
					tag = Tag(id=tag_row['id'], name=tag_row['name'], description=tag_row['description'])
					matches.append((tag, score))

		# Sort by score
		matches.sort(key=lambda x: x[1], reverse=True)
		return matches

	def get_tag_statistics(self, tag_name: str) -> dict:
		"""
		Get statistics about a tag's usage.

		Args:
		    tag_name: Name of the tag

		Returns:
		    Dictionary with statistics
		"""
		cursor = self.database.conn.cursor()

		# Get tag
		cursor.execute('SELECT id FROM tags WHERE name = ?', (tag_name,))
		row = cursor.fetchone()

		if not row:
			return {'error': f"Tag '{tag_name}' not found"}

		tag_id = row['id']

		# Count items
		cursor.execute(
			'''
            SELECT COUNT(*) as count FROM item_tags WHERE tag_id = ?
        ''',
			(tag_id,),
		)
		item_count = cursor.fetchone()['count']

		# Count selectors
		cursor.execute(
			'''
            SELECT COUNT(*) as count FROM tag_selectors WHERE tag_id = ?
        ''',
			(tag_id,),
		)
		selector_count = cursor.fetchone()['count']

		# Get average confidence
		cursor.execute(
			'''
            SELECT AVG(confidence) as avg_conf FROM tag_selectors WHERE tag_id = ?
        ''',
			(tag_id,),
		)
		avg_confidence = cursor.fetchone()['avg_conf'] or 0.0

		# Get page distribution
		cursor.execute(
			'''
            SELECT i.page_url, COUNT(*) as count
            FROM items i
            JOIN item_tags it ON i.id = it.item_id
            WHERE it.tag_id = ?
            GROUP BY i.page_url
            ORDER BY count DESC
            LIMIT 5
        ''',
			(tag_id,),
		)
		top_pages = [{'url': row['page_url'], 'count': row['count']} for row in cursor.fetchall()]

		return {
			'tag_name': tag_name,
			'item_count': item_count,
			'selector_count': selector_count,
			'average_confidence': round(avg_confidence, 3),
			'top_pages': top_pages,
		}

	def get_all_tags(self) -> list[Tag]:
		"""Get all tags in the database."""
		cursor = self.database.conn.cursor()
		cursor.execute('SELECT * FROM tags ORDER BY name')

		return [
			Tag(id=row['id'], name=row['name'], description=row['description'], created_at=row['created_at'])
			for row in cursor.fetchall()
		]
