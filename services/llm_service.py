import os
from dotenv import load_dotenv
from decouple import config
from haystack_integrations.document_stores.weaviate.document_store import (
    WeaviateDocumentStore,
)
from haystack_integrations.components.retrievers.weaviate import (
    WeaviateEmbeddingRetriever,
)
from haystack_integrations.components.generators.google_genai import (
    GoogleGenAIChatGenerator,
)
from haystack_integrations.components.embedders.google_genai import (
    GoogleGenAIDocumentEmbedder,
    GoogleGenAITextEmbedder,
)

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    OpenAITextEmbedder,
)
from haystack.utils import Secret
from utils.logger import logger


class LLMService:
    """Provides shared instances of LLMs and Weaviate vector store components."""

    def __init__(self):
        load_dotenv()

        doc_dir = str(config("DOC_DIR"))
        png_dir = str(config("PNG_DIR"))

        os.makedirs(doc_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        self._model_provider = config("MODEL_PROVIDER")
        self._model_name = config("MODEL_NAME")
        self._embedding_model_name = config("EMBEDDING_MODEL")
        self._llm = self.model_selection()
        self._docmument_embedder = self.model_selection_doc_embedder()
        self._text_embedder = self.model_selection_text_embedder()

    # model selection
    def model_selection(self):
        if self._model_provider == "openai":
            return self.open_ai_chat()
        elif self._model_provider == "gemini":
            return self.gemini_chat()

    def model_selection_doc_embedder(self):
        if self._model_provider == "openai":
            return self.open_ai_doc_embedder()
        elif self._model_provider == "gemini":
            return self.gemini_doc_embedder()

    def model_selection_text_embedder(self):
        if self._model_provider == "openai":
            return self.open_ai_text_embedder()
        elif self._model_provider == "gemini":
            return self.gemini_text_embedder()

    def open_ai_chat(self) -> OpenAIChatGenerator:
        """Returns an OpenAI Chat Generator instance."""
        return OpenAIChatGenerator(
            model=self._model_name,
        )

    def gemini_chat(self) -> GoogleGenAIChatGenerator:
        """Returns a Gemini Chat Generator instance."""
        return GoogleGenAIChatGenerator(
            model=self._model_name,
        )

    def open_ai_doc_embedder(self) -> OpenAIDocumentEmbedder:
        """Returns an OpenAI Document Embedder instance."""
        return OpenAIDocumentEmbedder(
            model=self._embedding_model_name,
        )

    def gemini_doc_embedder(self) -> GoogleGenAIDocumentEmbedder:
        """Returns a Gemini Document Embedder instance."""
        return GoogleGenAIDocumentEmbedder(
            model=self._embedding_model_name,
        )

    def open_ai_text_embedder(self) -> OpenAITextEmbedder:
        """Returns an OpenAI Text Embedder instance."""
        return OpenAITextEmbedder(
            model=self._embedding_model_name,
        )

    def gemini_text_embedder(self) -> GoogleGenAITextEmbedder:
        """Returns a Gemini Text Embedder instance."""
        return GoogleGenAITextEmbedder(
            model=self._embedding_model_name,
        )

    def weaviate_store(self) -> WeaviateDocumentStore:
        """Returns a WeaviateDocumentStore instance."""
        url = os.getenv("WEAVIATE_URL")
        return WeaviateDocumentStore(url=url)

    def weaviate_retriever(self, top_k: int = 3) -> WeaviateEmbeddingRetriever:
        """Returns a Weaviate embedding retriever."""
        return WeaviateEmbeddingRetriever(
            document_store=self.weaviate_store(), top_k=top_k
        )
