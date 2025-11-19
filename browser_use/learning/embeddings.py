"""
Embedding generation for text and images.

Supports:
- Text embeddings via sentence-transformers or OpenAI
- Image embeddings via CLIP or similar models
"""

import logging
from typing import Any

try:
	from sentence_transformers import SentenceTransformer

	SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
	SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
	from PIL import Image

	PIL_AVAILABLE = True
except ImportError:
	PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingService:
	"""Service for generating embeddings."""

	def __init__(
		self, text_model_name: str = 'all-MiniLM-L6-v2', image_model_name: str | None = None, use_openai: bool = False
	):
		"""
		Initialize embedding service.

		Args:
		    text_model_name: Name of the sentence-transformers model for text
		    image_model_name: Name of the model for images (optional)
		    use_openai: Use OpenAI embeddings instead of local models
		"""
		self.use_openai = use_openai
		self.text_model = None
		self.image_model = None

		if not use_openai and SENTENCE_TRANSFORMERS_AVAILABLE:
			try:
				self.text_model = SentenceTransformer(text_model_name)
				logger.info(f'Loaded text embedding model: {text_model_name}')
			except Exception as e:
				logger.warning(f'Failed to load text model: {e}')
		elif use_openai:
			logger.info('Using OpenAI embeddings')
		else:
			logger.warning('sentence-transformers not available. Install with: pip install sentence-transformers')

		# Image embeddings (CLIP) - optional for now
		if image_model_name and PIL_AVAILABLE:
			try:
				# Try to load CLIP if available
				import clip
				import torch

				self.image_model, _ = clip.load(image_model_name, device='cpu')
				logger.info(f'Loaded image embedding model: {image_model_name}')
			except ImportError:
				logger.warning('CLIP not available for image embeddings. Install with: pip install clip')
			except Exception as e:
				logger.warning(f'Failed to load image model: {e}')

	def embed_text(self, text: str) -> list[float] | None:
		"""
		Generate embedding for text.

		Args:
		    text: Text to embed

		Returns:
		    Embedding vector as list of floats, or None if failed
		"""
		if not text:
			return None

		try:
			if self.use_openai:
				return self._embed_text_openai(text)
			elif self.text_model:
				embedding = self.text_model.encode(text, convert_to_numpy=True)
				return embedding.tolist()
			else:
				logger.warning('No text embedding model available')
				return None
		except Exception as e:
			logger.error(f'Failed to generate text embedding: {e}')
			return None

	def _embed_text_openai(self, text: str) -> list[float] | None:
		"""Generate embedding using OpenAI API."""
		try:
			import openai

			# Use text-embedding-3-small for cost-effectiveness
			response = openai.embeddings.create(input=text, model='text-embedding-3-small')
			return response.data[0].embedding
		except Exception as e:
			logger.error(f'OpenAI embedding failed: {e}')
			return None

	def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
		"""
		Generate embeddings for multiple texts.

		Args:
		    texts: List of texts to embed

		Returns:
		    List of embedding vectors
		"""
		if not texts:
			return []

		try:
			if self.use_openai:
				return self._embed_text_batch_openai(texts)
			elif self.text_model:
				embeddings = self.text_model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 10)
				return embeddings.tolist()
			else:
				logger.warning('No text embedding model available')
				return []
		except Exception as e:
			logger.error(f'Failed to generate batch embeddings: {e}')
			return []

	def _embed_text_batch_openai(self, texts: list[str]) -> list[list[float]]:
		"""Generate embeddings using OpenAI API in batch."""
		try:
			import openai

			response = openai.embeddings.create(input=texts, model='text-embedding-3-small')
			return [item.embedding for item in response.data]
		except Exception as e:
			logger.error(f'OpenAI batch embedding failed: {e}')
			return []

	def embed_image(self, image_path: str) -> list[float] | None:
		"""
		Generate embedding for an image.

		Args:
		    image_path: Path to image file

		Returns:
		    Embedding vector as list of floats, or None if failed
		"""
		if not self.image_model or not PIL_AVAILABLE:
			logger.warning('Image embedding not available')
			return None

		try:
			import torch

			image = Image.open(image_path).convert('RGB')

			# Preprocess and encode
			image_input = self.image_model.preprocess(image).unsqueeze(0)
			with torch.no_grad():
				image_features = self.image_model.encode_image(image_input)
				embedding = image_features.squeeze().cpu().numpy()

			return embedding.tolist()
		except Exception as e:
			logger.error(f'Failed to generate image embedding: {e}')
			return None

	def get_embedding_dimension(self) -> int:
		"""Get the dimension of embeddings produced by this service."""
		if self.text_model:
			return self.text_model.get_sentence_embedding_dimension()
		elif self.use_openai:
			return 1536  # text-embedding-3-small dimension
		else:
			return 384  # Default for all-MiniLM-L6-v2


class SimpleEmbeddingService:
	"""Simplified embedding service that doesn't require external models."""

	def __init__(self):
		"""Initialize simple embedding service using basic text features."""
		self.logger = logger
		logger.info('Using simple embedding service (no external models)')

	def embed_text(self, text: str) -> list[float] | None:
		"""
		Generate a simple embedding based on text features.

		This is a fallback that doesn't require any external models.
		It's not as good as proper embeddings but can work for basic matching.

		Args:
		    text: Text to embed

		Returns:
		    Simple feature vector
		"""
		if not text:
			return None

		# Create a simple 64-dimensional feature vector
		# based on text characteristics
		import hashlib

		features = [0.0] * 64

		# Text length features
		features[0] = min(len(text) / 100.0, 1.0)
		features[1] = len(text.split()) / 20.0

		# Character distribution
		features[2] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
		features[3] = sum(1 for c in text if c.islower()) / max(len(text), 1)
		features[4] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
		features[5] = sum(1 for c in text if c.isspace()) / max(len(text), 1)

		# Hash-based features for semantic similarity
		# (not perfect but gives some differentiation)
		text_lower = text.lower()
		for i, word in enumerate(text_lower.split()[:10]):
			hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
			idx = 6 + (i * 5)
			if idx < 64:
				features[idx] = (hash_val % 1000) / 1000.0

		return features

	def embed_text_batch(self, texts: list[str]) -> list[list[float]]:
		"""Generate simple embeddings for multiple texts."""
		return [self.embed_text(text) or [0.0] * 64 for text in texts]

	def get_embedding_dimension(self) -> int:
		"""Get dimension of embeddings."""
		return 64
