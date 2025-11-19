"""
Advanced example showing Agent integration with learning.

This demonstrates:
1. Extracting items from multiple pages
2. Learning patterns and storing knowledge
3. Using learned knowledge for navigation
4. Feedback loop for improving selector confidence
"""

import asyncio

from browser_use import Browser
from browser_use.learning import LearningService


async def learn_from_multiple_pages():
	"""Visit multiple pages and learn from them."""
	browser = Browser()

	try:
		async with await browser.new_session() as session:
			learning_service = LearningService(browser_session=session, use_vision=False)

			# Pages to learn from
			pages = [
				('https://github.com/browser-use/browser-use', 'github_repo'),
				('https://github.com/browser-use/browser-use/issues', 'github_issues'),
			]

			for url, tag in pages:
				print(f'\nLearning from: {url}')
				page = session.page
				await page.goto(url, timeout=30000)

				# Extract and tag
				items = await learning_service.extract_and_learn(interactive=False, auto_tag=tag, max_items=15)
				print(f'  Learned {len(items)} items tagged as "{tag}"')

			# Show what we've learned
			print('\n' + '=' * 80)
			print('LEARNED KNOWLEDGE SUMMARY')
			print('=' * 80)

			for tag in ['github_repo', 'github_issues']:
				stats = learning_service.get_tag_statistics(tag)
				print(f'\nTag: {tag}')
				print(f'  Items: {stats["item_count"]}')
				print(f'  Selectors: {stats["selector_count"]}')

				# Show best selectors
				results = learning_service.query_by_tag(tag)
				print('  Top selectors:')
				for sel in results.get('selectors', [])[:3]:
					print(f'    - {sel["selector"]} (confidence: {sel["confidence"]:.2f})')

			learning_service.close()

	finally:
		await browser.close()


async def use_learned_knowledge():
	"""Demonstrate using learned knowledge for navigation."""
	browser = Browser()

	try:
		async with await browser.new_session() as session:
			learning_service = LearningService(browser_session=session, use_vision=False)

			# Navigate to a page
			page = session.page
			await page.goto('https://github.com/browser-use/browser-use', timeout=30000)

			# Query for specific items
			print('\nSearching for items similar to "star repository"...')
			similar = learning_service.search_similar('star repository', top_k=5)

			if similar:
				print(f'Found {len(similar)} similar items:')
				for result in similar:
					print(
						f'  - {result.item.cleaned_text or result.item.raw_text} '
						f'(score: {result.score:.2f}, selector: {result.item.selector})'
					)
			else:
				print('No similar items found (no embeddings available)')

			# Simulate learning from an action
			print('\nSimulating successful action...')
			await learning_service.learn_from_action(
				tag_name='star_button',
				selector='#star-button',  # Example selector
				success=True,
				page_url=page.url,
			)

			print('✓ Learned from action')

			learning_service.close()

	finally:
		await browser.close()


async def main():
	print('LEARNING LOOP DEMONSTRATION')
	print('=' * 80)

	print('\n[Step 1] Learning from multiple pages...')
	await learn_from_multiple_pages()

	print('\n[Step 2] Using learned knowledge...')
	await use_learned_knowledge()

	print('\n' + '=' * 80)
	print('DEMONSTRATION COMPLETE')
	print('=' * 80)


if __name__ == '__main__':
	asyncio.run(main())
