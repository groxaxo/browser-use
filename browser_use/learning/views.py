"""
Data models for the learning module.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ItemType(str, Enum):
	"""Types of items that can be extracted from a page."""

	LINK = 'link'
	BUTTON = 'button'
	FIELD = 'field'
	PRODUCT = 'product'
	IMAGE = 'image'
	HEADING = 'heading'
	OTHER = 'other'


class BoundingBox(BaseModel):
	"""Bounding box coordinates for an element."""

	x: float
	y: float
	width: float
	height: float


class ItemCandidate(BaseModel):
	"""A candidate item extracted from a page (before storage)."""

	# Core identification
	selector: str | None = None
	dom_path: str | None = None
	page_url: str
	page_title: str | None = None

	# Content
	raw_text: str | None = None
	cleaned_text: str | None = None
	item_type: ItemType = ItemType.OTHER

	# Position and visual
	bbox: BoundingBox | None = None
	screenshot_path: str | None = None

	# Embeddings (computed later)
	text_embedding: list[float] | None = None
	image_embedding: list[float] | None = None

	# Metadata
	attributes: dict[str, Any] = Field(default_factory=dict)
	confidence_score: float = 1.0

	# LM Reader outputs
	summary: str | None = None
	keywords: list[str] = Field(default_factory=list)


class StoredItem(BaseModel):
	"""An item stored in the database."""

	id: int
	page_url: str
	page_title: str | None = None
	selector: str | None = None
	dom_path: str | None = None
	raw_text: str | None = None
	cleaned_text: str | None = None
	item_type: ItemType = ItemType.OTHER
	bbox: str | None = None  # JSON string
	screenshot_path: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)

	# Vector indices (if stored in FAISS)
	text_vector_id: int | None = None
	image_vector_id: int | None = None


class Tag(BaseModel):
	"""A user-defined tag for items."""

	id: int | None = None
	name: str
	description: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)


class ItemTag(BaseModel):
	"""Association between an item and a tag."""

	item_id: int
	tag_id: int
	user_id: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)


class SelectorConfidence(BaseModel):
	"""Confidence score for a selector associated with a tag."""

	id: int | None = None
	tag_id: int
	selector: str
	confidence: float = 0.0  # How often this selector matched the user's selection
	usage_count: int = 0
	success_count: int = 0
	last_used: datetime = Field(default_factory=datetime.utcnow)


class TagEmbedding(BaseModel):
	"""Centroid embedding for a tag (average of all items with that tag)."""

	tag_id: int
	embedding: list[float]
	item_count: int = 0
	last_updated: datetime = Field(default_factory=datetime.utcnow)


class QueryResult(BaseModel):
	"""Result from querying the database."""

	item: StoredItem
	score: float  # Similarity or confidence score
	tags: list[Tag] = Field(default_factory=list)
