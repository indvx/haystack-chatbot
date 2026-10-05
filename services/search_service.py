from typing import Dict, Any, Tuple
from haystack.dataclasses import ChatMessage

from services.llm_service import LLMService
from tools.rag_search_tool import rag_tool
from utils.logger import logger


class SearchService:
    """Handles query execution across Haystack RAG search with LLM answer generation."""

    def __init__(self):
        self.__llm_service = LLMService()
        self.generator = self.__llm_service._llm

        self.system_prompt = (
            "You are an intelligent AI assistant.\n"
            "- Answer user questions using relevant document context when provided.\n"
            "- If context contains relevant information, synthesize a clear, accurate, and concise answer.\n"
            "- If no relevant information is found, answer using general knowledge or state clearly what is missing."
        )

    def _get_tool_result(self, query: str) -> Tuple[str, str]:
        """Attempts RAG semantic search over indexed Weaviate documents."""
        try:
            rag_result = rag_tool.invoke(text=query)
            documents = rag_result.get("documents", [])
            if documents and len(documents) > 0:
                text = "\n\n".join(
                    [
                        doc.content
                        for doc in documents
                        if hasattr(doc, "content") and doc.content
                    ]
                )
                if text.strip():
                    return text, "RAG"
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")

        return "No relevant information found in documents.", "None"

    def search(self, query: str) -> Dict[str, Any]:
        """Executes RAG retrieval and LLM reasoning to generate an answer."""
        logger.info(f"Searching query: '{query}'")
        tool_output, tool_used = self._get_tool_result(query)

        try:
            user_content = f"User Question: {query}\n\nRetrieved Context:\n{tool_output}" if tool_used == "RAG" else query
            messages = [
                ChatMessage.from_system(self.system_prompt),
                ChatMessage.from_user(user_content),
            ]
            response = self.generator.run(messages=messages)
            if response and "replies" in response and len(response["replies"]) > 0:
                answer = response["replies"][0].text
            else:
                answer = tool_output
        except Exception as e:
            logger.error(f"LLM execution error: {e}")
            answer = tool_output if tool_output != "No relevant information found in documents." else "Unable to process query."

        return {"query": query, "answer": answer, "tool_used": tool_used}
