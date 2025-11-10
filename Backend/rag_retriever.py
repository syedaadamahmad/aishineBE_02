# """
# RAG Retriever
# Performs semantic search with metadata filtering and context ranking.
# """
# import logging
# from typing import List, Dict, Any, Optional
# from Backend.embedding_client import BedrockEmbeddingClient
# from Backend.mongodb_client import MongoDBClient


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class RAGRetriever:
#     """Semantic retrieval with metadata filtering and ranking."""
    
#     def __init__(self):
#         self.embedding_client = BedrockEmbeddingClient()
#         self.mongo_client = MongoDBClient()
#         self.similarity_threshold = 0.75
#         self.max_results = 5
    
#     def extract_keywords(self, query: str) -> List[str]:
#         """
#         Extract substantive keywords from query.
#         Simple implementation: split and filter stop words.
#         """
#         stop_words = {
#             'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
#             'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
#             'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
#             'could', 'may', 'might', 'can', 'what', 'when', 'where', 'who', 'how'
#         }
        
#         words = query.lower().split()
#         keywords = [w.strip('.,!?;:') for w in words if w.lower() not in stop_words and len(w) > 2]
#         logger.info(f"[KEYWORDS] Extracted: {keywords}")
#         return keywords
    
#     def retrieve(
#         self,
#         query: str,
#         metadata_filters: Optional[Dict[str, Any]] = None
#     ) -> Dict[str, Any]:
#         """
#         Retrieve relevant context for a query.
        
#         Args:
#             query: User query string
#             metadata_filters: Optional filters (e.g., {"module_name": "module1_kb"})
        
#         Returns:
#             Dict with 'chunks', 'provenance', 'score_threshold_met'
#         """
#         try:
#             # Generate query embedding
#             logger.info(f"[RETRIEVE] Query: {query}")
#             query_embedding = self.embedding_client.generate_embedding(query)
            
#             if not query_embedding:
#                 logger.error("[RETRIEVE_ERR] Failed to generate query embedding")
#                 return {
#                     "chunks": [],
#                     "provenance": [],
#                     "score_threshold_met": False
#                 }
            
#             # Perform vector search
#             results = self.mongo_client.vector_search(
#                 query_embedding=query_embedding,
#                 limit=self.max_results,
#                 similarity_threshold=self.similarity_threshold,
#                 metadata_filters=metadata_filters
#             )
            
#             if not results:
#                 logger.warning("[RETRIEVE] No results above similarity threshold")
#                 return {
#                     "chunks": [],
#                     "provenance": [],
#                     "score_threshold_met": False
#                 }
            
#             # Format results
#             chunks = []
#             provenance = []
            
#             for idx, doc in enumerate(results):
#                 chunk_text = f"Topic: {doc.get('topic', 'N/A')}\n"
#                 chunk_text += f"Category: {doc.get('category', 'N/A')}\n"
#                 chunk_text += f"Summary: {doc.get('summary', 'N/A')}"
                
#                 chunks.append(chunk_text)
#                 provenance.append({
#                     "doc_id": str(doc.get('_id', '')),
#                     "topic": doc.get('topic', 'N/A'),
#                     "score": round(doc.get('score', 0.0), 3),
#                     "module": doc.get('module_name', 'N/A')
#                 })
            
#             logger.info(f"[RETRIEVE_OK] Retrieved {len(chunks)} chunks")
#             return {
#                 "chunks": chunks,
#                 "provenance": provenance,
#                 "score_threshold_met": True
#             }
            
#         except Exception as e:
#             logger.error(f"[RETRIEVE_ERR] {e}")
#             return {
#                 "chunks": [],
#                 "provenance": [],
#                 "score_threshold_met": False
#             }
    
#     def retrieve_for_continuation(self, previous_topic: str) -> Dict[str, Any]:
#         """
#         Retrieve additional context for continuation requests.
#         Uses previous topic/context to find related chunks.
#         """
#         logger.info(f"[CONTINUATION_RETRIEVE] Previous topic: {previous_topic}")
#         return self.retrieve(query=previous_topic)


"""
RAG Retriever
Performs semantic search with metadata filtering and context ranking.
"""
import logging
from typing import List, Dict, Any, Optional
from Backend.embedding_client import BedrockEmbeddingClient
from Backend.mongodb_client import MongoDBClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetriever:
    """Semantic retrieval with metadata filtering and ranking."""
    
    def __init__(self):
        self.embedding_client = BedrockEmbeddingClient()
        self.mongo_client = MongoDBClient()
        self.similarity_threshold = 0.75
        self.max_results = 5
    
    def retrieve(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query string
            metadata_filters: Optional filters (e.g., {"module_name": "module1_kb"})
        
        Returns:
            Dict with 'chunks', 'provenance', 'score_threshold_met'
        """
        try:
            logger.info(f"[RETRIEVE] Query: {query}")
            query_embedding = self.embedding_client.generate_embedding(query)
            
            if not query_embedding:
                logger.error("[RETRIEVE_ERR] Failed to generate query embedding")
                return {
                    "chunks": [],
                    "provenance": [],
                    "score_threshold_met": False
                }
            
            results = self.mongo_client.vector_search(
                query_embedding=query_embedding,
                limit=self.max_results,
                similarity_threshold=self.similarity_threshold,
                metadata_filters=metadata_filters
            )
            
            if not results:
                logger.warning("[RETRIEVE] No results above similarity threshold")
                return {
                    "chunks": [],
                    "provenance": [],
                    "score_threshold_met": False
                }
            
            chunks = []
            provenance = []
            
            for idx, doc in enumerate(results):
                # Use full content if available, else summary
                content = doc.get('content', '') or doc.get('summary', '')
                
                chunk_text = f"Topic: {doc.get('topic', 'N/A')}\n"
                chunk_text += f"Category: {doc.get('category', 'N/A')}\n"
                chunk_text += f"Content: {content}"
                
                chunks.append(chunk_text)
                provenance.append({
                    "doc_id": str(doc.get('_id', '')),
                    "topic": doc.get('topic', 'N/A'),
                    "score": round(doc.get('score', 0.0), 3),
                    "module": doc.get('module_name', 'N/A')
                })
            
            logger.info(f"[RETRIEVE_OK] Retrieved {len(chunks)} chunks")
            return {
                "chunks": chunks,
                "provenance": provenance,
                "score_threshold_met": True
            }
            
        except Exception as e:
            logger.error(f"[RETRIEVE_ERR] {e}")
            return {
                "chunks": [],
                "provenance": [],
                "score_threshold_met": False
            }