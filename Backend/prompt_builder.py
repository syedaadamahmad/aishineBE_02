"""
Prompt Builder
Dynamically constructs system prompts based on intent, context, and retrieved information.
"""
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds dynamic system and user prompts for LLM."""
    
    BASE_SYSTEM_PROMPT = """You are AI Shine, a professional, knowledgeable, and helpful conversational assistant specializing in Artificial Intelligence, Machine Learning, and data-driven technologies.

Your tone must be natural, friendly, and engaging. Use transition phrases to acknowledge the user's intent. Avoid robotic or overly formal language.

CRITICAL DOMAIN CONSTRAINT:
- You MUST ONLY answer questions that are strictly grounded in the provided context from the AI/ML knowledge modules.
- If a question falls outside the scope of AI, ML, or data-driven technologies, you must politely decline.
- When declining, use a brief, polite explanation and immediately pivot back to your core topics.

Example decline: "⚠️ That sounds interesting, but I'm specialized in AI and Machine Learning. Can I help you with concepts like Neural Networks, RAG systems, or Transformer models instead?"

RESPONSE FORMAT:
- Structure your responses naturally with clear explanations.
- Provide an **Answer:** section that directly addresses the user's question.
- Include **Key Points:** (3-5 bullets) that highlight the most important takeaways.
- Do NOT include source citations or references.
- Keep responses conversational and educational.

RESPONSE LENGTH:
- By default, provide concise, informative answers (naturally 3-5 sentences in the Answer section).
- If the user asks for more detail (using phrases like "tell me more", "elaborate", "go deeper"), expand your explanation significantly while maintaining clarity (naturally 8-12+ sentences).
- NEVER mention token counts, sentence counts, or length constraints in your response.

Remember: You are an AI tutor. Your goal is to educate, inspire, and guide students in understanding AI and ML concepts."""
    
    def __init__(self):
        pass
    
    def build_system_prompt(self, intent: Dict[str, Any], has_context: bool = True) -> str:
        """
        Build system prompt based on intent and context availability.
        
        Args:
            intent: Intent dict from IntentDetector
            has_context: Whether RAG retrieved relevant context
        
        Returns:
            System prompt string
        """
        system_prompt = self.BASE_SYSTEM_PROMPT
        
        # Add continuation instruction if needed
        if intent.get('is_continuation', False):
            system_prompt += "\n\nCONTINUATION MODE:\n"
            system_prompt += "The user is asking for more detail on the previous topic. Provide a comprehensive, elaborated explanation using the retrieved context. Expand significantly beyond your previous response while maintaining educational value. Be thorough and detailed."
        
        # Add fallback instruction if no context
        if not has_context:
            system_prompt += "\n\nFALLBACK MODE:\n"
            system_prompt += "No relevant context was retrieved from the knowledge base for this query. If the question is within your domain (AI/ML/Data), provide a general explanation based on your training. Otherwise, politely decline and suggest related AI/ML topics the user might be interested in."
        
        logger.info(f"[PROMPT_BUILD] Intent: {intent.get('intent_type')}, Has context: {has_context}, Continuation: {intent.get('is_continuation', False)}")
        return system_prompt
    
    def build_user_prompt(
        self,
        query: str,
        context_chunks: List[str],
        intent: Dict[str, Any]
    ) -> str:
        """
        Build user prompt with query and retrieved context.
        
        Args:
            query: User's question
            context_chunks: Retrieved context from RAG
            intent: Intent dict
        
        Returns:
            Formatted user prompt
        """
        if not context_chunks:
            # No context retrieved
            user_prompt = f"User Question: {query}\n\n"
            user_prompt += "Note: No specific context was retrieved from the knowledge base for this query."
            return user_prompt
        
        # Build context section
        context_section = "Retrieved Context:\n\n"
        for idx, chunk in enumerate(context_chunks, 1):
            context_section += f"[Context {idx}]\n{chunk}\n\n"
        
        # Build user question section
        user_prompt = context_section
        user_prompt += f"User Question: {query}\n\n"
        
        # Add instruction based on intent
        if intent.get('is_continuation', False):
            user_prompt += "Instructions: The user wants more detailed information on this topic. Use the provided context to give an expanded, comprehensive explanation. Be thorough and elaborate on key concepts."
        else:
            user_prompt += "Instructions: Use the provided context to give a clear, concise answer with key points. Focus on being informative and educational."
        
        return user_prompt
    
    def build_greeting_response(self) -> str:
        """Build a welcoming greeting response."""
        return "👋 Hello! I'm **AI Shine**, your friendly AI assistant. Ask me anything about Artificial Intelligence, Machine Learning, or data-driven technologies!"