from groq import AsyncGroq
from app.core.config import Config

class LLMClient:
    """
    Wrapper for Groq LLM API
    
    Responsibility: ONLY talk to external API
    Does NOT know about: RAG, retrieval, business logic
    """
    
    def __init__(self):
        self.config = Config()
        
        # Check API key exists
        if not self.config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found!")
        
        # Initialize Groq client
        self.client = AsyncGroq(api_key=self.config.GROQ_API_KEY)
        print(f" LLM Client initialized with model: {self.config.LLM_MODEL}")
    
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate text from LLM
        
        Args:
            prompt: The actual question/prompt
            system_prompt: Instructions for how LLM should behave
            
        Returns:
            Generated text response
        """
        # Build messages in format Groq expects
        messages = []
        
        # Add system instructions (optional)
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Call Groq API
        response = await self.client.chat.completions.create(
            model=self.config.LLM_MODEL,
            messages=messages,
            temperature=self.config.LLM_TEMPERATURE,
            max_tokens=self.config.LLM_MAX_TOKENS
        )
        
        # Extract and return text
        return response.choices[0].message.content