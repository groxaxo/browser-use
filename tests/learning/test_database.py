'''Unit tests for the database service.'''

import tempfile
from pathlib import Path

import pytest

from browser_use.learning.database import DatabaseService
from browser_use.learning.views import BoundingBox, ItemCandidate, ItemType


@pytest.fixture
def temp_db():
    '''Create a temporary database for testing.'''
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = DatabaseService(db_path=db_path, vector_dim=64)
    yield db
    
    db.close()
    Path(db_path).unlink(missing_ok=True)
    # Clean up FAISS files
    for ext in ['.faiss', '_map.json']:
        for prefix in ['text', 'image']:
            p = Path(db_path).parent / f'{Path(db_path).stem}_{prefix}{ext}'
            p.unlink(missing_ok=True)


def test_database_initialization(temp_db):
    '''Test database initialization.'''
    assert temp_db.conn is not None
    cursor = temp_db.conn.cursor()
    
    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    
    assert 'items' in tables
    assert 'tags' in tables
    assert 'item_tags' in tables


def test_add_item(temp_db):
    '''Test adding an item to the database.'''
    item = ItemCandidate(
        page_url='https://example.com',
        page_title='Example Page',
        selector='#test-button',
        raw_text='Click Me',
        cleaned_text='Click Me Button',
        item_type=ItemType.BUTTON,
        bbox=BoundingBox(x=10, y=20, width=100, height=50),
    )
    
    item_id = temp_db.add_item(item)
    assert item_id > 0
    
    # Retrieve item
    retrieved = temp_db.get_item(item_id)
    assert retrieved is not None
    assert retrieved.page_url == 'https://example.com'
    assert retrieved.selector == '#test-button'
    assert retrieved.cleaned_text == 'Click Me Button'


def test_tag_operations(temp_db):
    '''Test tag creation and association.'''
    # Create tag
    tag = temp_db.get_or_create_tag('test_tag', 'A test tag')
    assert tag.id is not None
    assert tag.name == 'test_tag'
    
    # Get same tag again
    tag2 = temp_db.get_or_create_tag('test_tag')
    assert tag2.id == tag.id
    
    # Add item
    item = ItemCandidate(
        page_url='https://example.com',
        raw_text='Test',
        item_type=ItemType.OTHER,
    )
    item_id = temp_db.add_item(item)
    
    # Tag item
    temp_db.tag_item(item_id, tag.id)
    
    # Retrieve tags for item
    tags = temp_db.get_item_tags(item_id)
    assert len(tags) == 1
    assert tags[0].name == 'test_tag'


def test_selector_confidence(temp_db):
    '''Test selector confidence tracking.'''
    tag = temp_db.get_or_create_tag('button_tag')
    selector = '#my-button'
    
    # Record successful use
    temp_db.update_selector_confidence(tag.id, selector, success=True)
    
    # Get best selectors
    best = temp_db.get_best_selectors_for_tag(tag.id, limit=5)
    assert len(best) == 1
    assert best[0].selector == selector
    assert best[0].confidence == 1.0
    assert best[0].usage_count == 1
    
    # Record failure
    temp_db.update_selector_confidence(tag.id, selector, success=False)
    
    best = temp_db.get_best_selectors_for_tag(tag.id, limit=5)
    assert best[0].confidence == 0.5  # 1 success out of 2 attempts
    assert best[0].usage_count == 2


def test_get_items_by_tag(temp_db):
    '''Test retrieving items by tag.'''
    tag = temp_db.get_or_create_tag('product')
    
    # Add multiple items
    for i in range(3):
        item = ItemCandidate(
            page_url=f'https://example.com/product{i}',
            raw_text=f'Product {i}',
            item_type=ItemType.PRODUCT,
        )
        item_id = temp_db.add_item(item)
        temp_db.tag_item(item_id, tag.id)
    
    # Retrieve
    items = temp_db.get_items_by_tag('product')
    assert len(items) == 3
