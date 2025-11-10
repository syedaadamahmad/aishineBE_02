"""
Intent Detector
Identifies continuation cues, greetings, and other conversational patterns.
"""
import re
import logging
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentDetector:
    """Detects user intent patterns from chat messages."""
    
    # Continuation patterns
    CONTINUATION_PATTERNS = [
        r'\btell\s+me\s+more\b',
        r'\belaborate\b',
        r'\bgo\s+deeper\b',
        r'\bexpand\s+on\b',
        r'\bmore\s+detail',
        r'\bcontinue\b',
        r'\bkeep\s+going\b',
        r'\bwhat\s+else\b',
        r'\bcan\s+you\s+explain\s+further\b',
        r'\btell\s+me\s+about.*more\b'
    ]
    
    # Greeting patterns
    GREETING_PATTERNS = [
        r'^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|sup|yo)\s*[!.,]?\s*$'
    ]
    
    def __init__(self):
        self.continuation_regex = re.compile('|'.join(self.CONTINUATION_PATTERNS), re.IGNORECASE)
        self.greeting_regex = re.compile('|'.join(self.GREETING_PATTERNS), re.IGNORECASE)
    
    def detect(self, message: str, chat_history: list = None) -> Dict[str, Any]:
        """
        Detect intent from user message.
        
        Args:
            message: Current user message
            chat_history: Previous messages (for context)
        
        Returns:
            Dict with 'intent_type', 'is_continuation', 'is_greeting', 'confidence'
        """
        message = message.strip()
        
        # Check for greeting
        is_greeting = bool(self.greeting_regex.match(message))
        if is_greeting:
            logger.info("[INTENT] Greeting detected")
            return {
                "intent_type": "greeting",
                "is_continuation": False,
                "is_greeting": True,
                "confidence": 1.0
            }
        
        # Check for continuation cues
        is_continuation = bool(self.continuation_regex.search(message))
        if is_continuation:
            logger.info("[INTENT] Continuation cue detected")
            return {
                "intent_type": "continuation",
                "is_continuation": True,
                "is_greeting": False,
                "confidence": 1.0
            }
        
        # Default: standard query
        logger.info("[INTENT] Standard query")
        return {
            "intent_type": "query",
            "is_continuation": False,
            "is_greeting": False,
            "confidence": 1.0
        }
