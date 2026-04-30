import os
from langchain_groq import ChatGroq


def load_llm():
    """
    Load LLaMA 3 model via Groq API.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Set it using:\n"
            'export GROQ_API_KEY="your_key_here"'
        )

    print("Loading LLaMA 3 (Groq)...")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",  
        temperature=0.2,
        max_tokens=512
    )

    print("LLaMA 3 loaded successfully")

    return llm