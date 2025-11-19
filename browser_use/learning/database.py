"""
Database service for storing and retrieving learned items.

Uses SQLite for structured data and FAISS for vector search.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

try:
	import faiss
	import numpy as np

	FAISS_AVAILABLE = True
except ImportError:
	FAISS_AVAILABLE = False

from browser_use.learning.views import (
	ItemCandidate,
	ItemTag,
	ItemType,
	QueryResult,
	SelectorConfidence,
	StoredItem,
	Tag,
	TagEmbedding,
)

logger = logging.getLogger(__name__)


class DatabaseService:
	"""Service for managing the learning database."""

	def __init__(self, db_path: str = 'browser_use_learning.db', vector_dim: int = 384):
		"""
		Initialize the database service.

		Args:
		    db_path: Path to SQLite database file
		    vector_dim: Dimension of embedding vectors (default 384 for sentence-transformers)
		"""
		self.db_path = Path(db_path)
		self.vector_dim = vector_dim
		self.conn: sqlite3.Connection | None = None

		# FAISS indices
		self.text_index: Any | None = None
		self.image_index: Any | None = None
		self.text_id_map: dict[int, int] = {}  # vector_id -> item_id
		self.image_id_map: dict[int, int] = {}  # vector_id -> item_id
		self.next_text_vector_id = 0
		self.next_image_vector_id = 0

		# Initialize
		self._initialize_database()
		if FAISS_AVAILABLE:
			self._initialize_vector_indices()
		else:
			logger.warning('FAISS not available. Vector search will be disabled. Install with: pip install faiss-cpu')

	def _initialize_database(self) -> None:
		"""Create database tables if they don't exist."""
		self.conn = sqlite3.connect(str(self.db_path), isolation_level=None)
		self.conn.row_factory = sqlite3.Row

		# Enable WAL mode for better concurrency
		self.conn.execute('PRAGMA journal_mode=WAL')

		# Create tables
		cursor = self.conn.cursor()

		# Items table
		cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_url TEXT NOT NULL,
                page_title TEXT,
                selector TEXT,
                dom_path TEXT,
                raw_text TEXT,
                cleaned_text TEXT,
                item_type TEXT DEFAULT 'other',
                bbox TEXT,
                screenshot_path TEXT,
                text_vector_id INTEGER,
                image_vector_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

		# Tags table
		cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

		# Item-Tag association
		cursor.execute('''
            CREATE TABLE IF NOT EXISTS item_tags (
                item_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (item_id, tag_id),
                FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
        ''')

		# Item metadata (key-value for extensibility)
		cursor.execute('''
            CREATE TABLE IF NOT EXISTS item_meta (
                item_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (item_id, key),
                FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE
            )
        ''')

		# Selector confidence tracking
		cursor.execute('''
            CREATE TABLE IF NOT EXISTS tag_selectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_id INTEGER NOT NULL,
                selector TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                UNIQUE(tag_id, selector)
            )
        ''')

		# Tag embeddings (centroids)
		cursor.execute('''
            CREATE TABLE IF NOT EXISTS tag_embeddings (
                tag_id INTEGER PRIMARY KEY,
                embedding TEXT NOT NULL,
                item_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
        ''')

		# Create indices for performance
		cursor.execute('CREATE INDEX IF NOT EXISTS idx_items_page_url ON items(page_url)')
		cursor.execute('CREATE INDEX IF NOT EXISTS idx_items_type ON items(item_type)')
		cursor.execute('CREATE INDEX IF NOT EXISTS idx_item_tags_tag ON item_tags(tag_id)')
		cursor.execute('CREATE INDEX IF NOT EXISTS idx_item_tags_item ON item_tags(item_id)')
		cursor.execute('CREATE INDEX IF NOT EXISTS idx_tag_selectors_tag ON tag_selectors(tag_id)')

		# Enable full-text search on cleaned_text
		try:
			cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
                    cleaned_text,
                    content=items,
                    content_rowid=id
                )
            ''')
		except sqlite3.OperationalError:
			logger.warning('FTS5 not available, full-text search will be limited')

		self.conn.commit()
		logger.info(f'Database initialized at {self.db_path}')

	def _initialize_vector_indices(self) -> None:
		"""Initialize FAISS indices for vector search."""
		if not FAISS_AVAILABLE:
			return

		# Use IndexFlatL2 for simplicity (can upgrade to HNSW for larger datasets)
		self.text_index = faiss.IndexFlatL2(self.vector_dim)
		self.image_index = faiss.IndexFlatL2(self.vector_dim)

		# Try to load existing indices
		index_path = self.db_path.parent / f'{self.db_path.stem}_text.faiss'
		id_map_path = self.db_path.parent / f'{self.db_path.stem}_text_map.json'

		if index_path.exists() and id_map_path.exists():
			try:
				self.text_index = faiss.read_index(str(index_path))
				with open(id_map_path, 'r') as f:
					map_data = json.load(f)
					self.text_id_map = {int(k): v for k, v in map_data['id_map'].items()}
					self.next_text_vector_id = map_data['next_id']
				logger.info(f'Loaded text index with {self.text_index.ntotal} vectors')
			except Exception as e:
				logger.warning(f'Failed to load text index: {e}')

		# Load image index
		image_index_path = self.db_path.parent / f'{self.db_path.stem}_image.faiss'
		image_id_map_path = self.db_path.parent / f'{self.db_path.stem}_image_map.json'

		if image_index_path.exists() and image_id_map_path.exists():
			try:
				self.image_index = faiss.read_index(str(image_index_path))
				with open(image_id_map_path, 'r') as f:
					map_data = json.load(f)
					self.image_id_map = {int(k): v for k, v in map_data['id_map'].items()}
					self.next_image_vector_id = map_data['next_id']
				logger.info(f'Loaded image index with {self.image_index.ntotal} vectors')
			except Exception as e:
				logger.warning(f'Failed to load image index: {e}')

	def save_indices(self) -> None:
		"""Save FAISS indices to disk."""
		if not FAISS_AVAILABLE or self.text_index is None:
			return

		try:
			# Save text index
			index_path = self.db_path.parent / f'{self.db_path.stem}_text.faiss'
			id_map_path = self.db_path.parent / f'{self.db_path.stem}_text_map.json'

			faiss.write_index(self.text_index, str(index_path))
			with open(id_map_path, 'w') as f:
				json.dump({'id_map': {k: v for k, v in self.text_id_map.items()}, 'next_id': self.next_text_vector_id}, f)

			# Save image index
			image_index_path = self.db_path.parent / f'{self.db_path.stem}_image.faiss'
			image_id_map_path = self.db_path.parent / f'{self.db_path.stem}_image_map.json'

			faiss.write_index(self.image_index, str(image_index_path))
			with open(image_id_map_path, 'w') as f:
				json.dump(
					{'id_map': {k: v for k, v in self.image_id_map.items()}, 'next_id': self.next_image_vector_id}, f
				)

			logger.info('Saved vector indices')
		except Exception as e:
			logger.error(f'Failed to save indices: {e}')

	def add_item(self, item: ItemCandidate) -> int:
		"""
		Add an item to the database.

		Args:
		    item: ItemCandidate to store

		Returns:
		    Item ID
		"""
		cursor = self.conn.cursor()

		# Convert bbox to JSON if present
		bbox_json = json.dumps(item.bbox.model_dump()) if item.bbox else None

		cursor.execute(
			'''
            INSERT INTO items (
                page_url, page_title, selector, dom_path,
                raw_text, cleaned_text, item_type, bbox, screenshot_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
			(
				item.page_url,
				item.page_title,
				item.selector,
				item.dom_path,
				item.raw_text,
				item.cleaned_text,
				item.item_type.value if item.item_type else ItemType.OTHER.value,
				bbox_json,
				item.screenshot_path,
			),
		)

		item_id = cursor.lastrowid

		# Add to vector indices if embeddings present
		if FAISS_AVAILABLE and item.text_embedding:
			self._add_text_embedding(item_id, item.text_embedding)

		if FAISS_AVAILABLE and item.image_embedding:
			self._add_image_embedding(item_id, item.image_embedding)

		# Store metadata
		for key, value in item.attributes.items():
			cursor.execute(
				'INSERT INTO item_meta (item_id, key, value) VALUES (?, ?, ?)', (item_id, key, str(value))
			)

		self.conn.commit()
		return item_id

	def _add_text_embedding(self, item_id: int, embedding: list[float]) -> None:
		"""Add text embedding to FAISS index."""
		if not FAISS_AVAILABLE or self.text_index is None:
			return

		vector = np.array([embedding], dtype='float32')
		vector_id = self.next_text_vector_id
		self.text_index.add(vector)
		self.text_id_map[vector_id] = item_id
		self.next_text_vector_id += 1

		# Update item with vector_id
		self.conn.execute('UPDATE items SET text_vector_id = ? WHERE id = ?', (vector_id, item_id))

	def _add_image_embedding(self, item_id: int, embedding: list[float]) -> None:
		"""Add image embedding to FAISS index."""
		if not FAISS_AVAILABLE or self.image_index is None:
			return

		vector = np.array([embedding], dtype='float32')
		vector_id = self.next_image_vector_id
		self.image_index.add(vector)
		self.image_id_map[vector_id] = item_id
		self.next_image_vector_id += 1

		# Update item with vector_id
		self.conn.execute('UPDATE items SET image_vector_id = ? WHERE id = ?', (vector_id, item_id))

	def get_or_create_tag(self, name: str, description: str | None = None) -> Tag:
		"""Get existing tag or create new one."""
		cursor = self.conn.cursor()

		# Try to get existing
		cursor.execute('SELECT * FROM tags WHERE name = ?', (name,))
		row = cursor.fetchone()

		if row:
			return Tag(id=row['id'], name=row['name'], description=row['description'], created_at=row['created_at'])

		# Create new
		cursor.execute('INSERT INTO tags (name, description) VALUES (?, ?)', (name, description))
		tag_id = cursor.lastrowid
		self.conn.commit()

		return Tag(id=tag_id, name=name, description=description)

	def tag_item(self, item_id: int, tag_id: int, user_id: str | None = None) -> None:
		"""Associate a tag with an item."""
		cursor = self.conn.cursor()
		try:
			cursor.execute('INSERT INTO item_tags (item_id, tag_id, user_id) VALUES (?, ?, ?)', (item_id, tag_id, user_id))
			self.conn.commit()

			# Update tag embedding after tagging
			self._update_tag_embedding(tag_id)
		except sqlite3.IntegrityError:
			# Already tagged
			pass

	def _update_tag_embedding(self, tag_id: int) -> None:
		"""Update the centroid embedding for a tag."""
		if not FAISS_AVAILABLE:
			return

		# Get all items with this tag that have text embeddings
		cursor = self.conn.cursor()
		cursor.execute(
			'''
            SELECT i.id, i.text_vector_id
            FROM items i
            JOIN item_tags it ON i.id = it.item_id
            WHERE it.tag_id = ? AND i.text_vector_id IS NOT NULL
        ''',
			(tag_id,),
		)

		rows = cursor.fetchall()
		if not rows:
			return

		# Compute centroid
		embeddings = []
		for row in rows:
			vector_id = row['text_vector_id']
			if vector_id in self.text_id_map:
				# Reconstruct vector from FAISS
				vector = self.text_index.reconstruct(int(vector_id))
				embeddings.append(vector)

		if embeddings:
			centroid = np.mean(embeddings, axis=0).tolist()
			cursor.execute(
				'''
                INSERT OR REPLACE INTO tag_embeddings (tag_id, embedding, item_count, last_updated)
                VALUES (?, ?, ?, ?)
            ''',
				(tag_id, json.dumps(centroid), len(embeddings), datetime.utcnow()),
			)
			self.conn.commit()

	def search_by_embedding(self, embedding: list[float], top_k: int = 5, use_text: bool = True) -> list[QueryResult]:
		"""
		Search for similar items using vector similarity.

		Args:
		    embedding: Query embedding vector
		    top_k: Number of results to return
		    use_text: Use text index (True) or image index (False)

		Returns:
		    List of QueryResult objects
		"""
		if not FAISS_AVAILABLE:
			logger.warning('FAISS not available, cannot perform vector search')
			return []

		index = self.text_index if use_text else self.image_index
		id_map = self.text_id_map if use_text else self.image_id_map

		if index is None or index.ntotal == 0:
			return []

		# Search
		query_vector = np.array([embedding], dtype='float32')
		distances, indices = index.search(query_vector, min(top_k, index.ntotal))

		results = []
		for dist, idx in zip(distances[0], indices[0]):
			if idx == -1:  # No result
				continue

			item_id = id_map.get(int(idx))
			if item_id is None:
				continue

			item = self.get_item(item_id)
			if item:
				# Convert distance to similarity score (lower is better in L2)
				score = 1.0 / (1.0 + float(dist))
				tags = self.get_item_tags(item_id)
				results.append(QueryResult(item=item, score=score, tags=tags))

		return results

	def get_item(self, item_id: int) -> StoredItem | None:
		"""Get an item by ID."""
		cursor = self.conn.cursor()
		cursor.execute('SELECT * FROM items WHERE id = ?', (item_id,))
		row = cursor.fetchone()

		if not row:
			return None

		return StoredItem(
			id=row['id'],
			page_url=row['page_url'],
			page_title=row['page_title'],
			selector=row['selector'],
			dom_path=row['dom_path'],
			raw_text=row['raw_text'],
			cleaned_text=row['cleaned_text'],
			item_type=ItemType(row['item_type']) if row['item_type'] else ItemType.OTHER,
			bbox=row['bbox'],
			screenshot_path=row['screenshot_path'],
			text_vector_id=row['text_vector_id'],
			image_vector_id=row['image_vector_id'],
			created_at=row['created_at'],
		)

	def get_item_tags(self, item_id: int) -> list[Tag]:
		"""Get all tags for an item."""
		cursor = self.conn.cursor()
		cursor.execute(
			'''
            SELECT t.* FROM tags t
            JOIN item_tags it ON t.id = it.tag_id
            WHERE it.item_id = ?
        ''',
			(item_id,),
		)

		return [
			Tag(id=row['id'], name=row['name'], description=row['description'], created_at=row['created_at'])
			for row in cursor.fetchall()
		]

	def get_items_by_tag(self, tag_name: str) -> list[StoredItem]:
		"""Get all items with a specific tag."""
		cursor = self.conn.cursor()
		cursor.execute(
			'''
            SELECT i.* FROM items i
            JOIN item_tags it ON i.id = it.item_id
            JOIN tags t ON it.tag_id = t.id
            WHERE t.name = ?
        ''',
			(tag_name,),
		)

		items = []
		for row in cursor.fetchall():
			items.append(
				StoredItem(
					id=row['id'],
					page_url=row['page_url'],
					page_title=row['page_title'],
					selector=row['selector'],
					dom_path=row['dom_path'],
					raw_text=row['raw_text'],
					cleaned_text=row['cleaned_text'],
					item_type=ItemType(row['item_type']) if row['item_type'] else ItemType.OTHER,
					bbox=row['bbox'],
					screenshot_path=row['screenshot_path'],
					text_vector_id=row['text_vector_id'],
					image_vector_id=row['image_vector_id'],
					created_at=row['created_at'],
				)
			)

		return items

	def update_selector_confidence(self, tag_id: int, selector: str, success: bool) -> None:
		"""Update confidence score for a selector."""
		cursor = self.conn.cursor()

		# Get or create selector entry
		cursor.execute(
			'SELECT * FROM tag_selectors WHERE tag_id = ? AND selector = ?',
			(tag_id, selector),
		)
		row = cursor.fetchone()

		if row:
			usage_count = row['usage_count'] + 1
			success_count = row['success_count'] + (1 if success else 0)
			confidence = success_count / usage_count

			cursor.execute(
				'''
                UPDATE tag_selectors
                SET usage_count = ?, success_count = ?, confidence = ?, last_used = ?
                WHERE id = ?
            ''',
				(usage_count, success_count, confidence, datetime.utcnow(), row['id']),
			)
		else:
			usage_count = 1
			success_count = 1 if success else 0
			confidence = success_count / usage_count

			cursor.execute(
				'''
                INSERT INTO tag_selectors (tag_id, selector, usage_count, success_count, confidence, last_used)
                VALUES (?, ?, ?, ?, ?, ?)
            ''',
				(tag_id, selector, usage_count, success_count, confidence, datetime.utcnow()),
			)

		self.conn.commit()

	def get_best_selectors_for_tag(self, tag_id: int, limit: int = 5) -> list[SelectorConfidence]:
		"""Get the best selectors for a tag, sorted by confidence."""
		cursor = self.conn.cursor()
		cursor.execute(
			'''
            SELECT * FROM tag_selectors
            WHERE tag_id = ?
            ORDER BY confidence DESC, usage_count DESC
            LIMIT ?
        ''',
			(tag_id, limit),
		)

		return [
			SelectorConfidence(
				id=row['id'],
				tag_id=row['tag_id'],
				selector=row['selector'],
				confidence=row['confidence'],
				usage_count=row['usage_count'],
				success_count=row['success_count'],
				last_used=row['last_used'],
			)
			for row in cursor.fetchall()
		]

	def close(self) -> None:
		"""Close database connection and save indices."""
		self.save_indices()
		if self.conn:
			self.conn.close()
			self.conn = None

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()
