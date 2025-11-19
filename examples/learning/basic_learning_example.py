# Basic example of using the learning loop

import asyncio

from browser_use import Browser
from browser_use.learning import LearningService


async def main():
	browser = Browser()
	try:
		async with await browser.new_session() as session:
			page = session.page
			await page.goto('https://github.com/browser-use/browser-use', timeout=30000)
			print('Navigated to GitHub page')

			learning_service = LearningService(browser_session=session, use_vision=False)
			print('Extracting items from page...')
			items = await learning_service.extract_and_learn(interactive=False, auto_tag='github_repo', max_items=20)

			print(f'Extracted and tagged {len(items)} items')
			learning_service.close()
	finally:
		await browser.close()


if __name__ == '__main__':
	asyncio.run(main())
