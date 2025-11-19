'''Unit tests for the retrieval service.'''

import tempfile
from pathlib import Path

import pytest

from browser_use.learning.database import DatabaseService
from browser_use.learning.retrieval import RetrievalService
from browser_use.learning.views import ItemCandidate, ItemType


@pytest.fixture
def temp_db():
    '''Create a temporary database for testing.'''
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = DatabaseService(db_path=db_path, vector_dim=64)
    yield db
    
    db.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def retrieval_service(temp_db):
    '''Create retrieval service with sample data.'''
    # Add some sample items
    tag = temp_db.get_or_create_tag('test_button')
    
    for i in range(3):
        item = ItemCandidate(
            page_url=f'https://example.com/page{i}',
            raw_text=f'Button {i}',
            cleaned_text=f'Click Button {i}',
            item_type=ItemType.BUTTON,
            selector=f'#button-{i}',
        )
        item_id = temp_db.add_item(item)
        temp_db.tag_item(item_id, tag.id)
        temp_db.update_selector_confidence(tag.id, item.selector, success=True)
    
    return RetrievalService(database=temp_db)


def test_get_items_by_tag(retrieval_service):
    '''Test retrieving items by tag.'''
    items = retrieval_service.get_items_by_tag('test_button')
    assert len(items) == 3
    assert all(item.item_type == ItemType.BUTTON for item in items)


def test_get_best_selectors(retrieval_service):
    '''Test getting best selectors for a tag.'''
    selectors = retrieval_service.get_best_selectors('test_button', limit=5)
    assert len(selectors) == 3
    assert all(sel.confidence == 1.0 for sel in selectors)


def test_recommend_actions_for_tag(retrieval_service):
    '''Test action recommendations.'''
    recommendations = retrieval_service.recommend_actions_for_tag('test_button')
    
    assert recommendations['tag'] == 'test_button'
    assert len(recommendations['selectors']) == 3
    assert all(sel['confidence'] == 1.0 for sel in recommendations['selectors'])


def test_get_tag_statistics(retrieval_service):
    '''Test tag statistics retrieval.'''
    stats = retrieval_service.get_tag_statistics('test_button')
    
    assert stats['tag_name'] == 'test_button'
    assert stats['item_count'] == 3
    assert stats['selector_count'] == 3
    assert stats['average_confidence'] == 1.0


def test_get_all_tags(retrieval_service):
    '''Test retrieving all tags.'''
    tags = retrieval_service.get_all_tags()
    assert len(tags) >= 1
    assert any(tag.name == 'test_button' for tag in tags)


def test_find_tag_for_text(retrieval_service):
    '''Test finding tags by text matching.'''
    matches = retrieval_service.find_tag_for_text('button', threshold=0.5)
    assert len(matches) > 0
    assert matches[0][0].name == 'test_button'
