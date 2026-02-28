from groq import AsyncGroq
from app.core.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)

class LLMClient:
    """Async Wrapper for Groq LLM API"""
    
    def __init__(self):
        self.config = Config()
        
        if not self.config.GROQ_API_KEY:
            logger.error("GROQ_API_KEY not found in environment variables!")
            raise ValueError("GROQ_API_KEY not found in environment variables!")
        
        self.client = AsyncGroq(api_key=self.config.GROQ_API_KEY)
        logger.info(f"LLM Client initialized with model: {self.config.LLM_MODEL}")
    
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text from LLM (async)"""
        logger.debug(f"Generating response for prompt length: {len(prompt)}")
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=messages,
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS
            )
            
            result = response.choices[0].message.content
            logger.info(f"Generated response length: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise