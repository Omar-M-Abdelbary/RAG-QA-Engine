# app/core/constants.py

class RAGConstants:
    """Constants for RAG system"""
    
    # System prompt for LLM
    SYSTEM_PROMPT = """You are a knowledgeable AI assistant that answers questions based on provided context.

INSTRUCTIONS:
- Answer using ONLY the information from the context
- If context is insufficient, say "I don't have enough information"
- Be concise and accurate
- Do not make up information

Answer clearly and helpfully."""
    
    # Template for creating RAG prompts
    RAG_PROMPT_TEMPLATE = """Based on the following context, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    # Messages
    NO_RESULTS_MESSAGE = "I couldn't find relevant information to answer this question."
    INSUFFICIENT_INFO_MESSAGE = "I don't have enough information to answer this question."