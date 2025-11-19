"""
Learning module for Browser-Use.

This module implements a learning loop that allows the agent to:
- Extract and consolidate DOM + vision data from web pages
- Store items with embeddings in a local database
- Enable user tagging and feedback
- Use stored knowledge for improved navigation
"""

from browser_use.learning.database import DatabaseService
from browser_use.learning.embeddings import EmbeddingService, SimpleEmbeddingService
from browser_use.learning.extractor import DOMExtractor, ItemConsolidator, VisionExtractor
from browser_use.learning.reader import LMReader
from browser_use.learning.retrieval import RetrievalService
from browser_use.learning.service import LearningService, browse_and_extract
from browser_use.learning.tagging_cli import SimplifiedTaggingInterface, TaggingCLI
from browser_use.learning.views import ItemCandidate, QueryResult, StoredItem, Tag

__all__ = [
	'DatabaseService',
	'DOMExtractor',
	'VisionExtractor',
	'ItemConsolidator',
	'LMReader',
	'RetrievalService',
	'LearningService',
	'browse_and_extract',
	'EmbeddingService',
	'SimpleEmbeddingService',
	'TaggingCLI',
	'SimplifiedTaggingInterface',
	'ItemCandidate',
	'StoredItem',
	'Tag',
	'QueryResult',
]
