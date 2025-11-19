# Learning Module

The learning module extends Browser-Use with a learning loop that enables the agent to:

- **Extract and consolidate** DOM and vision data from web pages
- **Store items** with embeddings in a local database (SQLite + FAISS)
- **Enable user tagging** and feedback for item identification
- **Use stored knowledge** for improved navigation and automation

## Features

### 1. Multi-Source Extraction
- **DOM Extraction**: Identifies interactive elements (buttons, links, forms) using existing Browser-Use DOM services
- **Vision Extraction** (optional): Uses OCR (Tesseract) to extract text from screenshots
- **Consolidation**: Merges DOM and vision items, deduplicates, and ranks by importance

### 2. Content Normalization
- **LM Reader**: Uses LLMs to clean and canonicalize extracted text
- **Keyword Extraction**: Identifies important tokens from content
- **Type Classification**: Categorizes items (button, link, field, etc.)

### 3. Vector Storage & Retrieval
- **SQLite Database**: Stores items, tags, selectors, and metadata
- **FAISS Index**: Enables fast semantic search with embeddings
- **Embedding Options**: 
  - Local models (sentence-transformers)
  - OpenAI embeddings (requires API key)
  - Simple fallback (no external dependencies)

### 4. User Tagging
- **Interactive CLI**: Present items to user for tagging
- **Programmatic Tagging**: Bulk tag items via API
- **Selector Confidence**: Track which selectors work best for each tag

### 5. Knowledge Retrieval
- **Tag-based search**: Find items by user-defined tags
- **Semantic search**: Find similar items using embeddings
- **Selector recommendation**: Get best selectors for a tag based on confidence

## Quick Start

```python
import asyncio
from browser_use import Browser
from browser_use.learning import LearningService

async def main():
    browser = Browser()
    async with await browser.new_session() as session:
        page = session.page
        await page.goto('https://example.com')
        
        # Create learning service
        learning = LearningService(
            browser_session=session,
            use_vision=False  # Set True to enable OCR
        )
        
        # Extract and tag items
        items = await learning.extract_and_learn(
            interactive=False,
            auto_tag='example_site',
            max_items=50
        )
        
        # Query learned knowledge
        results = learning.query_by_tag('example_site')
        print(f"Best selectors: {results['selectors']}")
        
        learning.close()
    await browser.close()

asyncio.run(main())
```

## Installation

### Core Dependencies (included in browser-use)
- `sqlite3` (built-in)
- `pydantic` (for data models)

### Optional Dependencies

For vector search (recommended):
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
```

For embeddings:
```bash
pip install sentence-transformers  # Local embeddings
# OR use OpenAI embeddings (requires OPENAI_API_KEY)
```

For vision/OCR:
```bash
pip install pytesseract pillow
# Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract
```

For CLIP image embeddings:
```bash
pip install git+https://github.com/openai/CLIP.git
```

## Architecture

### Database Schema

```sql
-- Core items table
CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    page_url TEXT,
    selector TEXT,
    cleaned_text TEXT,
    item_type TEXT,
    bbox TEXT,  -- JSON
    created_at TIMESTAMP
);

-- User tags
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);

-- Item-tag associations
CREATE TABLE item_tags (
    item_id INTEGER,
    tag_id INTEGER,
    PRIMARY KEY (item_id, tag_id)
);

-- Selector confidence tracking
CREATE TABLE tag_selectors (
    tag_id INTEGER,
    selector TEXT,
    confidence REAL,
    usage_count INTEGER,
    success_count INTEGER
);
```

### Component Flow

1. **Extraction** → DOMExtractor + VisionExtractor
2. **Consolidation** → ItemConsolidator (merge, dedupe, rank)
3. **Cleaning** → LMReader (normalize text with LLM)
4. **Embedding** → EmbeddingService (generate vectors)
5. **Storage** → DatabaseService (SQLite + FAISS)
6. **Tagging** → TaggingCLI or SimplifiedTaggingInterface
7. **Retrieval** → RetrievalService (query by tag or similarity)

## API Reference

### LearningService

Main orchestration service.

```python
learning = LearningService(
    browser_session: BrowserSession,
    llm: BaseChatModel | None = None,
    db_path: str = 'browser_use_learning.db',
    use_vision: bool = False,
    use_openai_embeddings: bool = False
)

# Extract and learn from current page
items = await learning.extract_and_learn(
    interactive=bool,  # Use CLI for tagging
    auto_tag=str,      # Auto-tag all items
    max_items=int      # Limit extraction
)

# Query by tag
results = learning.query_by_tag('tag_name')

# Semantic search
similar = learning.search_similar('query text', top_k=5)

# Learn from action feedback
await learning.learn_from_action(
    tag_name='button',
    selector='#submit',
    success=True
)
```

### DatabaseService

Low-level database operations.

```python
db = DatabaseService(db_path='data.db', vector_dim=384)

# Add item
item_id = db.add_item(item_candidate)

# Tag item
tag = db.get_or_create_tag('my_tag')
db.tag_item(item_id, tag.id)

# Update selector confidence
db.update_selector_confidence(tag.id, selector, success=True)

# Search by embedding
results = db.search_by_embedding(embedding, top_k=5)
```

## Examples

See `examples/learning/` for complete examples:

- `basic_learning_example.py` - Simple extraction and tagging
- More examples coming soon!

## Configuration

### Embedding Models

```python
# Local embeddings (default)
learning = LearningService(
    browser_session=session,
    use_openai_embeddings=False
)

# OpenAI embeddings (requires OPENAI_API_KEY)
learning = LearningService(
    browser_session=session,
    use_openai_embeddings=True
)
```

### Vision Extraction

```python
# Enable OCR
learning = LearningService(
    browser_session=session,
    use_vision=True
)
```

### LM Reader

```python
from browser_use.llm import ChatOpenAI

llm = ChatOpenAI(model='gpt-4')
learning = LearningService(
    browser_session=session,
    llm=llm  # Use for content cleaning
)
```

## Performance Tips

1. **Batch Processing**: Extract multiple pages and batch-tag
2. **Vector Index**: Use FAISS HNSW for large datasets
3. **SQLite Tuning**: WAL mode is enabled by default
4. **Embeddings**: Local models are faster but less accurate than OpenAI

## Limitations

- **Vision extraction** requires Tesseract installation
- **FAISS** is optional but highly recommended for vector search
- **LM Reader** works better with an LLM but has heuristic fallback
- **Cross-origin iframes** may not be fully extractable

## Future Enhancements

- [ ] Fine-tune local embedding models on user data
- [ ] Auto-suggest tags based on page content
- [ ] Incremental learning without full re-extraction
- [ ] Multi-user tagging support
- [ ] Export/import learned knowledge
- [ ] Integration with Agent for automatic knowledge use

## Contributing

Contributions welcome! Please see the main Browser-Use contributing guidelines.
