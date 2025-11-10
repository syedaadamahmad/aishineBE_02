"""
MongoDB Atlas Vector Search Client
Handles connection pooling, vector search, and metadata filtering.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import certifi
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB Atlas Vector Search client with connection pooling."""
    
    def __init__(self):
        self.uri = os.getenv("MONGO_DB_URI")
        self.db_name = os.getenv("DB_NAME", "aishine")
        self.collection_name = "module_vectors"
        
        if not self.uri:
            raise ValueError("[MONGO_ERR] MONGO_DB_URI not set in environment")
    
        self.client = MongoClient(
            self.uri,
            maxPoolSize=10,
            minPoolSize=2,
            serverSelectionTimeoutMS=10000,
            tls=True,
            tlsCAFile=certifi.where(),
            )
            # Test connection
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        logger.info(f"[MONGO_OK] Connected to {self.db_name}.{self.collection_name}")
    
    def vector_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.75,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search with optional metadata filtering.
        
        Args:
            query_embedding: 1024-dim embedding vector
            limit: Max results to return
            similarity_threshold: Minimum cosine similarity score
            metadata_filters: Optional dict of metadata filters (e.g., {"module_name": "module1_kb"})
        
        Returns:
            List of documents with score >= threshold, sorted by relevance
        """
        try:
            # Build aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "score": {"$gte": similarity_threshold}
                    }
                }
            ]
            
            # Add metadata filters if provided
            if metadata_filters:
                pipeline.append({"$match": metadata_filters})
            
            # Project only necessary fields
            pipeline.append({
                "$project": {
                    "_id": 1,
                    "topic": 1,
                    "category": 1,
                    "summary": 1,
                    "keywords": 1,
                    "module_name": 1,
                    "score": 1
                }
            })
            
            results = list(self.collection.aggregate(pipeline))
            
            logger.info(f"[VECTOR_SEARCH] Retrieved {len(results)} chunks above threshold {similarity_threshold}")
            return results
            
        except OperationFailure as e:
            logger.error(f"[VECTOR_SEARCH_ERR] {e}")
            return []
        except Exception as e:
            logger.error(f"[VECTOR_SEARCH_ERR] Unexpected error: {e}")
            return []
    
    def insert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Bulk insert documents with embeddings.
        
        Args:
            documents: List of dicts with 'embedding', 'topic', 'summary', 'keywords', 'module_name', etc.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("[INSERT] No documents to insert")
                return False
            
            result = self.collection.insert_many(documents, ordered=False)
            logger.info(f"[INSERT_OK] Inserted {len(result.inserted_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"[INSERT_ERR] {e}")
            return False
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("[MONGO_CLOSE] Connection closed")
