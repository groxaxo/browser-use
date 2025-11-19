# Integrating Learning with Browser-Use Agent

This guide shows how to integrate the learning module with Browser-Use agents for improved navigation and automation.

## Basic Integration Pattern

The learning module can work alongside the main Agent to:
1. **Learn from agent actions** - Record what the agent does and how successful it was
2. **Guide agent decisions** - Provide the agent with learned selectors and patterns
3. **Improve over time** - Build up knowledge that makes future tasks easier

## Usage Patterns

### Pattern 1: Pre-learning Phase

Learn from pages before running automated tasks:

```python
import asyncio
from browser_use import Browser
from browser_use.learning import LearningService

async def prelearn():
    browser = Browser()
    async with await browser.new_session() as session:
        learning = LearningService(browser_session=session)
        
        # Visit and learn from pages
        for url in ['https://example.com/page1', 'https://example.com/page2']:
            await session.page.goto(url)
            await learning.extract_and_learn(auto_tag='example_site')
        
        learning.close()
    await browser.close()

asyncio.run(prelearn())
```

### Pattern 2: Learning from Agent Actions

Track what the agent does and learn from it:

```python
from browser_use import Agent, Browser
from browser_use.learning import LearningService

async def agent_with_learning():
    browser = Browser()
    async with await browser.new_session() as session:
        learning = LearningService(browser_session=session)
        
        # Run agent task
        agent = Agent(
            task="Find and click the login button",
            browser=browser,
            # ... other config
        )
        
        history = await agent.run()
        
        # Learn from the agent's actions
        for step in history.steps():
            if step.action and step.action.selector:
                # Record successful actions
                await learning.learn_from_action(
                    tag_name='login_button',
                    selector=step.action.selector,
                    success=step.success,
                )
        
        learning.close()
    await browser.close()
```

### Pattern 3: Using Learned Knowledge

Query learned knowledge before taking actions:

```python
async def use_learned_knowledge():
    browser = Browser()
    async with await browser.new_session() as session:
        learning = LearningService(browser_session=session)
        
        await session.page.goto('https://example.com')
        
        # Get best known selectors for a task
        recommendations = learning.query_by_tag('login_button')
        
        if recommendations['selectors']:
            best_selector = recommendations['selectors'][0]['selector']
            print(f"Using learned selector: {best_selector}")
            
            # Try the learned selector
            try:
                element = await session.page.query_selector(best_selector)
                if element:
                    await element.click()
                    # Record success
                    tag = learning.database.get_or_create_tag('login_button')
                    learning.database.update_selector_confidence(
                        tag.id, best_selector, success=True
                    )
            except Exception:
                # Record failure
                tag = learning.database.get_or_create_tag('login_button')
                learning.database.update_selector_confidence(
                    tag.id, best_selector, success=False
                )
        
        learning.close()
    await browser.close()
```

### Pattern 4: Semantic Search for Similar Elements

Use embeddings to find similar elements:

```python
async def find_similar_elements():
    browser = Browser()
    async with await browser.new_session() as session:
        learning = LearningService(
            browser_session=session,
            use_openai_embeddings=True  # Better results with OpenAI
        )
        
        await session.page.goto('https://example.com')
        
        # Search for elements similar to a description
        results = learning.search_similar(
            "submit form button",
            top_k=5
        )
        
        for result in results:
            print(f"Found: {result.item.cleaned_text}")
            print(f"  Selector: {result.item.selector}")
            print(f"  Similarity: {result.score:.2f}")
        
        learning.close()
    await browser.close()
```

## Custom Tool Integration

You can create custom tools that leverage learned knowledge:

```python
from browser_use import Tools
from browser_use.learning import LearningService

tools = Tools()

@tools.action(
    description="Click an element that was previously learned and tagged"
)
async def click_learned_element(tag_name: str, session, learning: LearningService):
    '''Click element using learned knowledge.'''
    recommendations = learning.query_by_tag(tag_name)
    
    if not recommendations['selectors']:
        return f"No learned selectors for tag: {tag_name}"
    
    # Try selectors in order of confidence
    for sel_info in recommendations['selectors']:
        selector = sel_info['selector']
        try:
            element = await session.page.query_selector(selector)
            if element:
                await element.click()
                # Record success
                tag = learning.database.get_or_create_tag(tag_name)
                learning.database.update_selector_confidence(
                    tag.id, selector, success=True
                )
                return f"Successfully clicked element using selector: {selector}"
        except Exception:
            # Try next selector
            continue
    
    return f"All learned selectors failed for tag: {tag_name}"

# Use in agent
agent = Agent(
    task="Click the login button",
    browser=browser,
    tools=tools,
)
```

## Best Practices

### 1. Tag Naming Conventions

Use consistent, descriptive tag names:
- Good: `login_button`, `search_field`, `product_card`
- Bad: `btn1`, `thing`, `element`

### 2. Confidence Thresholds

When using learned selectors, consider confidence scores:

```python
recommendations = learning.query_by_tag('my_element')
high_confidence = [
    sel for sel in recommendations['selectors'] 
    if sel['confidence'] > 0.7
]
```

### 3. Fallback Strategies

Always have a fallback when learned selectors fail:

```python
# Try learned selector first
success = try_learned_selector(tag_name)

if not success:
    # Fall back to agent exploration
    agent_result = await agent.run(task="Find and click the element")
    
    # Learn from agent's success
    if agent_result.success:
        await learning.learn_from_action(...)
```

### 4. Periodic Cleanup

Clean up old or low-confidence selectors:

```python
# Remove selectors with very low confidence
cursor = learning.database.conn.cursor()
cursor.execute('''
    DELETE FROM tag_selectors 
    WHERE confidence < 0.2 AND usage_count > 10
''')
learning.database.conn.commit()
```

### 5. Multi-site Learning

When learning from multiple sites, use site-specific tags:

```python
await learning.extract_and_learn(auto_tag=f'login_button_{site_name}')
```

## Performance Considerations

### Embedding Generation

- **Local models**: Fast but requires installation
- **OpenAI**: More accurate but requires API calls
- **Simple fallback**: Always available but less accurate

```python
# For production with many items
learning = LearningService(
    browser_session=session,
    use_openai_embeddings=True,  # Better accuracy
    use_vision=False,  # Disable if not needed
)
```

### Database Location

Store the learning database persistently:

```python
learning = LearningService(
    browser_session=session,
    db_path='/path/to/persistent/knowledge.db'
)
```

### Batch Operations

When learning from many pages, use batch operations:

```python
all_items = []
for url in urls:
    await page.goto(url)
    items = await learning.dom_extractor.extract_items()
    all_items.extend(items)

# Clean and add in batch
cleaned = await learning.reader.batch_clean_items(all_items)
for item in cleaned:
    learning.database.add_item(item)
```

## Troubleshooting

### "FAISS not available"

Install FAISS for vector search:
```bash
pip install faiss-cpu  # or faiss-gpu
```

### "No embeddings generated"

Install sentence-transformers:
```bash
pip install sentence-transformers
```

Or use OpenAI:
```python
learning = LearningService(..., use_openai_embeddings=True)
```

### "Tesseract not found"

For vision extraction, install Tesseract:
- macOS: `brew install tesseract`
- Ubuntu: `apt-get install tesseract-ocr`
- Windows: Download from https://github.com/tesseract-ocr/tesseract

## Future Enhancements

Planned improvements to the learning module:

1. **Automatic agent integration** - Agent automatically uses learned knowledge
2. **Multi-user learning** - Share learned knowledge across users
3. **Export/import** - Share learned databases
4. **Fine-tuning** - Adapt embeddings to specific domains
5. **Active learning** - Agent asks for user feedback when uncertain

## Contributing

Contributions to improve the learning module are welcome! See the main Browser-Use contributing guidelines.
