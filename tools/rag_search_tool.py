from typing import List, Dict, Any
from haystack.tools import ComponentTool
from haystack import component
from haystack.dataclasses import Document
from services.llm_service import LLMService

utility_service = LLMService()


@component()
class RagSearcher:
    """Haystack component performing vector search against Weaviate."""

    def __init__(self):
        self.text_embedder = utility_service._text_embedder
        self.retriever = utility_service.weaviate_retriever(top_k=3)

    @component.output_types(documents=List[Document])
    def run(self, text: str) -> Dict[str, Any]:
        emb_out = self.text_embedder.run(text=text)
        docs_out = self.retriever.run(query_embedding=emb_out["embedding"])
        return {"documents": docs_out.get("documents", [])}


rag_tool = ComponentTool(
    component=RagSearcher(),
    name="rag_search",
    description="Semantic search over indexed documents.",
)
