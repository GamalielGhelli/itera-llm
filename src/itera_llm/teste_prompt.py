from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from itera_llm.model import criar_modelo


def few_shot(llm: ChatOpenAI) -> None:
    """Testa few-shot prompting com exemplos de pergunta e resposta."""
    mensagens = [
        HumanMessage(content="Qual é o primeiro dia da semana?"),
        AIMessage(content="Domingo."),
        HumanMessage(content="Qual é o terceiro dia da semana?"),
        AIMessage(content="Terça-feira."),
        HumanMessage(content="Qual é o último dia da semana?"),
    ]

    resposta = llm.invoke(mensagens)
    print("\n--- RESPOSTA COM FEW-SHOT ---")
    print(resposta.content)


def resposta_em_etapas(llm: ChatOpenAI) -> None:
    """Solicita uma solução com etapas verificáveis."""
    mensagens = [
        HumanMessage(
            content="""Resolva o problema abaixo.

Apresente:
1. As informações importantes
2. As etapas verificáveis da solução
3. A resposta final

Problema:
Uma loja oferece 20% de desconto em um produto de R$ 150.
Qual é o preço final?"""
        ),
    ]

    resposta = llm.invoke(mensagens)
    print("\n--- RESPOSTA EM ETAPAS ---")
    print(resposta.content)


def persona_specific(llm: ChatOpenAI) -> None:
    mensagens = [
        SystemMessage(
            content=(
                "Você é um professor de inteligência artificial. "
                "Explique assuntos técnicos de forma didática e simples."
            )
        ),
        HumanMessage(content="O que é HDFS? Para que ele serve no Hadoop?"),
    ]

    resposta = llm.invoke(mensagens)
    print("\n--- PROMPT COM PERSONA ---")
    print(resposta.content)


def main() -> None:
    """Executa exclusivamente os experimentos deste módulo."""
    llm = criar_modelo()

    few_shot(llm)
    persona_specific(llm)
    resposta_em_etapas(llm)


if __name__ == "__main__":
    main()
