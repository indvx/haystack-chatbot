import os
from pathlib import Path
from typing import List, Optional, Union
from haystack import Pipeline, component
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.converters import (
    TextFileToDocument,
    PyPDFToDocument,
    CSVToDocument,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from decouple import config
from services.llm_service import LLMService
from utils.logger import logger
from haystack.dataclasses import Document


class FileUploadService:
    """Handles parsing, embedding, and indexing document files into Weaviate."""

    def __init__(self):
        self.__llm_service = LLMService()
        self.__doc_directory = Path(config("DOC_DIR", default="./documents"))
        self.__png_directory = Path(config("PNG_DIR", default="./pngs"))

        os.makedirs(self.__doc_directory, exist_ok=True)
        os.makedirs(self.__png_directory, exist_ok=True)

    def upload_data(
        self, target_files: Optional[List[Union[str, Path]]] = None
    ) -> dict:
        """Processes document files through the Haystack indexing pipeline."""
        if target_files:
            file_paths = [Path(f) for f in target_files if os.path.isfile(f)]
        else:
            file_paths = [
                self.__doc_directory / name
                for name in os.listdir(self.__doc_directory)
                if os.path.isfile(self.__doc_directory / name)
            ]

        if not file_paths:
            logger.warning("No files found to index.")
            return {
                "status": "warning",
                "message": "No files found to index",
                "processed_count": 0,
            }

        logger.info(
            f"Indexing {len(file_paths)} document(s): {[p.name for p in file_paths]}"
        )

        document_embedder = self.__llm_service._docmument_embedder
        document_writer = DocumentWriter(
            document_store=self.__llm_service.weaviate_store(),
            policy=DuplicatePolicy.OVERWRITE,
        )
        document_splitter = DocumentSplitter(
            split_by="word", split_length=150, split_overlap=50
        )
        document_cleaner = DocumentCleaner()
        metadata_cleaner = MetadataCleaner()
        text_converter = TextFileToDocument()
        csv_converter = CSVToDocument()
        pdf_converter = PyPDFToDocument()
        joiner = DocumentJoiner()
        file_type_router = FileTypeRouter(
            mime_types=["text/plain", "application/pdf", "text/csv"]
        )

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("file_type_router", file_type_router)
        indexing_pipeline.add_component("text_converter", text_converter)
        indexing_pipeline.add_component("csv_converter", csv_converter)
        indexing_pipeline.add_component("pdf_converter", pdf_converter)
        indexing_pipeline.add_component("joiner", joiner)
        indexing_pipeline.add_component("cleaner", document_cleaner)
        indexing_pipeline.add_component("splitter", document_splitter)
        indexing_pipeline.add_component("metadata_cleaner", metadata_cleaner)
        indexing_pipeline.add_component("embedder", document_embedder)
        indexing_pipeline.add_component("writer", document_writer)

        indexing_pipeline.connect(
            "file_type_router.text/plain", "text_converter.sources"
        )
        indexing_pipeline.connect(
            "file_type_router.application/pdf", "pdf_converter.sources"
        )
        indexing_pipeline.connect("file_type_router.text/csv", "csv_converter.sources")

        indexing_pipeline.connect("text_converter", "joiner")
        indexing_pipeline.connect("pdf_converter", "joiner")
        indexing_pipeline.connect("csv_converter", "joiner")

        indexing_pipeline.connect("joiner", "cleaner")
        indexing_pipeline.connect("cleaner", "splitter")
        indexing_pipeline.connect("splitter.documents", "metadata_cleaner.documents")
        indexing_pipeline.connect("metadata_cleaner.documents", "embedder.documents")
        indexing_pipeline.connect("embedder", "writer")

        try:
            indexing_pipeline.draw(path=self.__png_directory / "file_uploader.png")
        except Exception as e:
            logger.warning(f"Could not render pipeline diagram: {e}")

        indexing_pipeline.run({"file_type_router": {"sources": file_paths}})
        logger.info(f"Indexed {len(file_paths)} document(s) successfully.")

        return {
            "status": "success",
            "message": f"Successfully indexed {len(file_paths)} file(s)",
            "processed_count": len(file_paths),
            "files": [p.name for p in file_paths],
        }


@component
class MetadataCleaner:

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        allowed = {
            "file_path",
            "page_number",
            "source_id",
            "split_id",
            "split_idx_start",
        }

        for document in documents:
            document.meta = {
                key: value for key, value in document.meta.items() if key in allowed
            }

        return {"documents": documents}
