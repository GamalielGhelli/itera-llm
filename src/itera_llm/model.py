from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def criar_modelo( *, reasoning_effort: str = "none", max_tokens: int = 300,
) -> ChatOpenAI:
    """Cria o modelo compartilhado pelas diferentes partes da aplicação."""
    load_dotenv()

    return ChatOpenAI(
        model="gpt-6-luna",
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
    )
