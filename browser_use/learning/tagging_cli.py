"""
CLI interface for tagging items.

Provides an interactive terminal UI for users to:
- View extracted items
- Tag items
- Create new tags
- Bulk operations
"""

import logging
from typing import TYPE_CHECKING

from browser_use.learning.database import DatabaseService
from browser_use.learning.views import ItemCandidate

if TYPE_CHECKING:
	pass

logger = logging.getLogger(__name__)


class TaggingCLI:
	"""CLI for interactive tagging of extracted items."""

	def __init__(self, database: DatabaseService):
		"""
		Initialize tagging CLI.

		Args:
		    database: DatabaseService instance
		"""
		self.database = database
		self.logger = logger

	def present_items_for_tagging(self, items: list[ItemCandidate]) -> list[tuple[ItemCandidate, list[str]]]:
		"""
		Present items to user for tagging.

		Args:
		    items: List of ItemCandidate objects

		Returns:
		    List of (ItemCandidate, tag_names) tuples
		"""
		print('\n' + '=' * 80)
		print('ITEM TAGGING SESSION')
		print('=' * 80)
		print(f'\nFound {len(items)} items to review.\n')

		# Get existing tags
		existing_tags = self.database.get_all_tags() if hasattr(self.database, 'get_all_tags') else []
		tag_names = [tag.name for tag in existing_tags]

		if tag_names:
			print('Existing tags:')
			for i, name in enumerate(tag_names, 1):
				print(f'  {i}. {name}')
			print()

		tagged_items = []

		for idx, item in enumerate(items, 1):
			print(f'\n--- Item {idx}/{len(items)} ---')
			print(f'Type: {item.item_type.value}')
			if item.cleaned_text:
				print(f'Text: {item.cleaned_text}')
			elif item.raw_text:
				print(f'Raw text: {item.raw_text[:100]}')
			if item.selector:
				print(f'Selector: {item.selector}')
			if item.bbox:
				print(f'Position: ({item.bbox.x:.0f}, {item.bbox.y:.0f})')

			# Get user input
			print('\nOptions:')
			print('  - Enter tag names (comma-separated)')
			print('  - Type "new: <tag_name>" to create a new tag')
			print('  - Type "skip" to skip this item')
			print('  - Type "done" to finish tagging')

			user_input = input('\nYour choice: ').strip()

			if user_input.lower() == 'done':
				break
			elif user_input.lower() == 'skip':
				continue
			elif user_input.lower().startswith('new:'):
				# Create new tag
				new_tag_name = user_input[4:].strip()
				if new_tag_name:
					tagged_items.append((item, [new_tag_name]))
					tag_names.append(new_tag_name)
					print(f"✓ Created new tag '{new_tag_name}'")
			elif user_input:
				# Parse comma-separated tags
				selected_tags = [t.strip() for t in user_input.split(',') if t.strip()]
				if selected_tags:
					tagged_items.append((item, selected_tags))
					print(f'✓ Tagged with: {", ".join(selected_tags)}')

		print(f'\n✓ Tagged {len(tagged_items)} items')
		return tagged_items

	def save_tagged_items(self, tagged_items: list[tuple[ItemCandidate, list[str]]]) -> None:
		"""
		Save tagged items to database.

		Args:
		    tagged_items: List of (ItemCandidate, tag_names) tuples
		"""
		print('\nSaving tagged items to database...')

		for item, tag_names in tagged_items:
			# Add item
			item_id = self.database.add_item(item)

			# Add tags
			for tag_name in tag_names:
				tag = self.database.get_or_create_tag(tag_name)
				self.database.tag_item(item_id, tag.id)

				# Update selector confidence (assuming success since user tagged it)
				if item.selector:
					self.database.update_selector_confidence(tag.id, item.selector, success=True)

		print(f'✓ Saved {len(tagged_items)} items with tags')

	def interactive_tagging_session(self, items: list[ItemCandidate]) -> None:
		"""
		Run a full interactive tagging session.

		Args:
		    items: List of items to tag
		"""
		if not items:
			print('No items to tag.')
			return

		# Present items
		tagged_items = self.present_items_for_tagging(items)

		if not tagged_items:
			print('No items were tagged.')
			return

		# Save to database
		self.save_tagged_items(tagged_items)

		print('\n' + '=' * 80)
		print('TAGGING SESSION COMPLETE')
		print('=' * 80)


class SimplifiedTaggingInterface:
	"""Simplified non-interactive tagging interface for programmatic use."""

	def __init__(self, database: DatabaseService):
		"""
		Initialize simplified interface.

		Args:
		    database: DatabaseService instance
		"""
		self.database = database

	def tag_items_bulk(self, items: list[ItemCandidate], tag_names: list[str]) -> None:
		"""
		Tag all items with the same tags.

		Args:
		    items: List of items to tag
		    tag_names: List of tag names to apply
		"""
		logger.info(f'Bulk tagging {len(items)} items with tags: {", ".join(tag_names)}')

		for item in items:
			# Add item
			item_id = self.database.add_item(item)

			# Add tags
			for tag_name in tag_names:
				tag = self.database.get_or_create_tag(tag_name)
				self.database.tag_item(item_id, tag.id)

				# Update selector confidence
				if item.selector:
					self.database.update_selector_confidence(tag.id, item.selector, success=True)

		logger.info(f'✓ Tagged {len(items)} items')

	def tag_item_with_feedback(self, item: ItemCandidate, tag_name: str, success: bool) -> None:
		"""
		Tag an item and provide feedback on selector quality.

		Args:
		    item: Item to tag
		    tag_name: Tag name
		    success: Whether the selector successfully identified the intended element
		"""
		# Add item
		item_id = self.database.add_item(item)

		# Add tag
		tag = self.database.get_or_create_tag(tag_name)
		self.database.tag_item(item_id, tag.id)

		# Update selector confidence with feedback
		if item.selector:
			self.database.update_selector_confidence(tag.id, item.selector, success=success)

		logger.info(f"Tagged item with '{tag_name}', selector success: {success}")
