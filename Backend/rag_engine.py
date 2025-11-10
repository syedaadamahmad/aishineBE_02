# """ works
# RAG Engine - Core Orchestration
# Integrates intent detection, retrieval, prompt construction, and LLM generation.
# """
# import logging
# from typing import List, Dict, Any

# from Backend.models import Message, IntentResult, RetrievalContext
# from Backend.intent_detector import IntentDetector
# from Backend.rag_retriever import RAGRetriever
# from Backend.prompt_builder import PromptBuilder
# from Backend.llm_client import GeminiClient
# from Backend.memory_manager import MemoryManager
# from dotenv import load_dotenv

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class RAGEngine:
#     """
#     RAG orchestrator for AI Shine Tutor.
    
#     Pipeline:
#     1. Intent Detection
#     2. Memory Management (short-term context)
#     3. RAG Retrieval (semantic search)
#     4. Prompt Construction (dynamic brief/detailed)
#     5. LLM Generation
#     6. Response Formatting
#     """
    
#     def __init__(self):
#         """Initialize all RAG components."""
#         try:
#             self.intent_detector = IntentDetector()
#             self.retriever = RAGRetriever()
#             self.prompt_builder = PromptBuilder()
#             self.llm_client = GeminiClient()
#             self.memory_manager = MemoryManager(short_term_window=5)
            
#             logger.info("[RAG_ENGINE] ✅ All components initialized")
#         except Exception as e:
#             logger.error(f"[RAG_ENGINE_ERR] Initialization failed: {e}")
#             raise
    
#     def process_query(
#         self,
#         query: str,
#         chat_history: List[Message]
#     ) -> Dict[str, Any]:
#         """
#         Main processing pipeline.
        
#         Args:
#             query: Current user query
#             chat_history: Full conversation history
        
#         Returns:
#             Dict with 'answer' (str) and 'type' (str)
#         """
#         try:
#             # Step 1: Intent Detection
#             logger.info(f"[RAG_ENGINE] Step 1: Intent Detection")
#             intent = self.intent_detector.detect(query, chat_history)
#             logger.info(f"[RAG_ENGINE] Intent: {intent['intent_type']}")
            
#             # Step 2: Handle Greeting
#             if intent['is_greeting']:
#                 logger.info("[RAG_ENGINE] Greeting detected, returning welcome message")
#                 return {
#                     "answer": self.prompt_builder.build_greeting_response(),
#                     "type": "greeting"
#                 }
            
#             # Step 3: Memory Management
#             logger.info(f"[RAG_ENGINE] Step 2: Memory Management")
#             short_term_history = self.memory_manager.get_short_term_context(chat_history)
#             formatted_history = self.memory_manager.format_for_llm(short_term_history)
#             logger.info(f"[RAG_ENGINE] Using last {len(short_term_history)} messages for context")
            
#             # Step 4: RAG Retrieval
#             logger.info(f"[RAG_ENGINE] Step 3: RAG Retrieval")
#             retrieval_result = self.retriever.retrieve(query)
            
#             has_context = retrieval_result["score_threshold_met"] and len(retrieval_result["chunks"]) > 0
#             logger.info(f"[RAG_ENGINE] Retrieved {len(retrieval_result['chunks'])} chunks, threshold met: {has_context}")
            
#             # Step 5: Prompt Construction
#             logger.info(f"[RAG_ENGINE] Step 4: Prompt Construction")
#             system_prompt = self.prompt_builder.build_system_prompt(intent, has_context)
#             user_prompt = self.prompt_builder.build_user_prompt(
#                 query=query,
#                 context_chunks=retrieval_result["chunks"],
#                 intent=intent
#             )
            
#             # Step 6: LLM Generation
#             logger.info(f"[RAG_ENGINE] Step 5: LLM Generation")
#             llm_response = self.llm_client.generate_response(
#                 system_prompt=system_prompt,
#                 user_prompt=user_prompt,
#                 chat_history=formatted_history
#             )
            
#             if not llm_response["success"]:
#                 logger.error(f"[RAG_ENGINE_ERR] LLM generation failed: {llm_response['error']}")
#                 return {
#                     "answer": "⚠️ I encountered an issue processing your question. Please try again.",
#                     "type": "text"
#                 }
            
#             raw_answer = llm_response["response"]
            
#             # Step 7: Response Classification
#             logger.info(f"[RAG_ENGINE] Step 6: Response Classification")
#             response_type = self._classify_response(raw_answer, has_context)
#             logger.info(f"[RAG_ENGINE] Response type: {response_type}")
            
#             return {
#                 "answer": raw_answer,
#                 "type": response_type
#             }
        
#         except Exception as e:
#             logger.error(f"[RAG_ENGINE_ERR] Pipeline failure: {e}", exc_info=True)
#             return {
#                 "answer": "⚠️ An unexpected error occurred. Please try your question again.",
#                 "type": "text"
#             }
    
#     def _classify_response(self, response: str, has_context: bool) -> str:
#         """
#         Classify response type for frontend rendering.
        
#         Args:
#             response: Raw LLM output
#             has_context: Whether RAG retrieval succeeded
        
#         Returns:
#             One of: "structured", "decline", "text"
#         """
#         # Check for decline pattern
#         if response.startswith("⚠") or "I'm specialized in AI and Machine Learning" in response:
#             return "decline"
        
#         # Check for structured format
#         if "**Answer:**" in response and "**Key Points:**" in response:
#             return "structured"
        
#         # Default to plain text
#         return "text"
    
#     def cleanup(self):
#         """Close all persistent connections."""
#         try:
#             self.retriever.mongo_client.close()
#             logger.info("[RAG_ENGINE] ✅ Cleanup complete")
#         except Exception as e:
#             logger.error(f"[RAG_ENGINE] Cleanup error: {e}")
# # ```

# # ---

# # ## Architecture Summary

# # ### Data Flow Through RAG Engine
# # ```
# # User Query
# #     ↓
# # ┌─────────────────────────────────────────────┐
# # │ RAGEngine.process_query()                   │
# # ├─────────────────────────────────────────────┤
# # │ 1. IntentDetector.detect()                  │
# # │    → greeting / continuation / query        │
# # │                                             │
# # │ 2. MemoryManager.get_short_term_context()  │
# # │    → Last 5 messages for LLM               │
# # │                                             │
# # │ 3. RAGRetriever.retrieve()                 │
# # │    → Semantic search + metadata filter     │
# # │    → Cosine similarity threshold (0.75)    │
# # │                                             │
# # │ 4. PromptBuilder.build_system_prompt()     │
# # │    → Dynamic brief/detailed instructions   │
# # │                                             │
# # │ 5. GeminiClient.generate_response()        │
# # │    → Temperature 0.75                      │
# # │    → Hybrid memory context                 │
# # │                                             │
# # │ 6. _classify_response()                    │
# # │    → greeting / text / structured / decline│
# # └─────────────────────────────────────────────┘
# #     ↓
# # Response {answer, type}
# # ```

# # ---

# # ### Component Responsibilities
# # ```
# # ┌──────────────────────┬─────────────────────────────────────┐
# # │ Component            │ Responsibility                      │
# # ├──────────────────────┼─────────────────────────────────────┤
# # │ IntentDetector       │ Pattern matching (greeting, "tell   │
# # │                      │ me more", standard query)           │
# # ├──────────────────────┼─────────────────────────────────────┤
# # │ MemoryManager        │ Sliding window (last 5 msgs) +      │
# # │                      │ format for Gemini API               │
# # ├──────────────────────┼─────────────────────────────────────┤
# # │ RAGRetriever         │ Bedrock embeddings + MongoDB vector │
# # │                      │ search + metadata filtering         │
# # ├──────────────────────┼─────────────────────────────────────┤
# # │ PromptBuilder        │ System prompt construction with     │
# # │                      │ dynamic verbosity control           │
# # ├──────────────────────┼─────────────────────────────────────┤
# # │ GeminiClient         │ LLM generation with temperature     │
# # │                      │ 0.75 and structured output          │
# # ├──────────────────────┼─────────────────────────────────────┤
# # │ RAGEngine            │ Orchestrates all components in      │
# # │                      │ correct order with error handling   │
# # └──────────────────────┴─────────────────────────────────────┘

























"""
RAG Engine - Core Orchestration
Integrates intent detection, retrieval, prompt construction, and LLM generation.
CRITICAL FIX: Only passes conversation history when continuation is detected.
"""
import logging
from typing import List, Dict, Any

from Backend.models import Message
from Backend.intent_detector import IntentDetector
from Backend.rag_retriever import RAGRetriever
from Backend.prompt_builder import PromptBuilder
from Backend.llm_client import GeminiClient
from Backend.memory_manager import MemoryManager
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG orchestrator for AI Shine Tutor.
    
    Pipeline:
    1. Intent Detection
    2. Memory Management (short-term context)
    3. RAG Retrieval (semantic search)
    4. Prompt Construction (dynamic brief/detailed)
    5. LLM Generation
    6. Response Formatting
    """
    
    def __init__(self):
        """Initialize all RAG components."""
        try:
            self.intent_detector = IntentDetector()
            self.retriever = RAGRetriever()
            self.prompt_builder = PromptBuilder()
            self.llm_client = GeminiClient()
            self.memory_manager = MemoryManager(short_term_window=3)
            
            logger.info("[RAG_ENGINE] ✅ All components initialized")
        except Exception as e:
            logger.error(f"[RAG_ENGINE_ERR] Initialization failed: {e}")
            raise
    
    def process_query(
        self,
        query: str,
        chat_history: List[Message]
    ) -> Dict[str, Any]:
        """
        Main processing pipeline.
        
        Args:
            query: Current user query
            chat_history: Full conversation history
        
        Returns:
            Dict with 'answer' (str) and 'type' (str)
        """
        try:
            # Step 1: Intent Detection
            logger.info(f"[RAG_ENGINE] Step 1: Intent Detection")
            intent = self.intent_detector.detect(query, chat_history)
            logger.info(f"[RAG_ENGINE] Intent: {intent['intent_type']}, Continuation: {intent['is_continuation']}")
            
            # Step 2: Handle Greeting
            if intent['is_greeting']:
                logger.info("[RAG_ENGINE] Greeting detected, returning welcome message")
                return {
                    "answer": self.prompt_builder.build_greeting_response(),
                    "type": "greeting"
                }
            
            # Step 3: Memory Management
            logger.info(f"[RAG_ENGINE] Step 2: Memory Management")
            short_term_history = self.memory_manager.get_short_term_context(chat_history)
            
            # ✅ FIX: Only format history if continuation
            formatted_history = self.memory_manager.format_for_llm(
                short_term_history,
                is_continuation=intent['is_continuation']
            )
            
            if formatted_history:
                logger.info(f"[RAG_ENGINE] Passing {len(formatted_history)} messages to LLM (continuation mode)")
            else:
                logger.info(f"[RAG_ENGINE] No history passed to LLM (saving tokens)")
            
            # Step 4: RAG Retrieval
            logger.info(f"[RAG_ENGINE] Step 3: RAG Retrieval")
            retrieval_result = self.retriever.retrieve(query)
            
            has_context = retrieval_result["score_threshold_met"] and len(retrieval_result["chunks"]) > 0
            logger.info(f"[RAG_ENGINE] Retrieved {len(retrieval_result['chunks'])} chunks, threshold met: {has_context}")
            
            # Step 5: Prompt Construction
            logger.info(f"[RAG_ENGINE] Step 4: Prompt Construction")
            system_prompt = self.prompt_builder.build_system_prompt(intent, has_context)
            user_prompt = self.prompt_builder.build_user_prompt(
                query=query,
                context_chunks=retrieval_result["chunks"],
                intent=intent
            )
            
            # Step 6: LLM Generation
            logger.info(f"[RAG_ENGINE] Step 5: LLM Generation")
            llm_response = self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                chat_history=formatted_history  # ✅ Now None for non-continuation
            )
            
            if not llm_response["success"]:
                logger.error(f"[RAG_ENGINE_ERR] LLM generation failed: {llm_response['error']}")
                return {
                    "answer": "⚠️ I encountered an issue processing your question. Please try again.",
                    "type": "text"
                }
            
            raw_answer = llm_response["response"]
            
            # Step 7: Response Classification
            logger.info(f"[RAG_ENGINE] Step 6: Response Classification")
            response_type = self._classify_response(raw_answer, has_context)
            logger.info(f"[RAG_ENGINE] Response type: {response_type}")
            
            return {
                "answer": raw_answer,
                "type": response_type
            }
        
        except Exception as e:
            logger.error(f"[RAG_ENGINE_ERR] Pipeline failure: {e}", exc_info=True)
            return {
                "answer": "⚠️ An unexpected error occurred. Please try your question again.",
                "type": "text"
            }
    
    def _classify_response(self, response: str, has_context: bool) -> str:
        """
        Classify response type for frontend rendering.
        
        Args:
            response: Raw LLM output
            has_context: Whether RAG retrieval succeeded
        
        Returns:
            One of: "structured", "decline", "text"
        """
        # Check for decline pattern
        if response.startswith("⚠") or "I'm specialized in AI and Machine Learning" in response:
            return "decline"
        
        # Check for structured format
        if "**Answer:**" in response and "**Key Points:**" in response:
            return "structured"
        
        # Default to plain text
        return "text"
    
    def cleanup(self):
        """Close all persistent connections."""
        try:
            self.retriever.mongo_client.close()
            logger.info("[RAG_ENGINE] ✅ Cleanup complete")
        except Exception as e:
            logger.error(f"[RAG_ENGINE] Cleanup error: {e}")