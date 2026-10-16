# itera-llm

Projeto de estudos com LangChain e modelos da OpenAI.

## Organização

O código reutilizável fica em módulos dentro de `src/itera_llm`. Cada módulo
executável pode definir sua própria função `main()` e o bloco:

```python
if __name__ == "__main__":
    main()
```

Isso permite executar um módulo de forma independente, sem depender de um
entry point do projeto:

```bash
uv run python -m itera_llm.teste_prompt
```

O projeto também oferece um atalho para esse módulo:

```bash
uv run itera-prompts
```

Para um módulo futuro chamado `rag.py`, por exemplo, basta executá-lo com:

```bash
uv run python -m itera_llm.rag
```

Um novo entry point só é necessário quando você quiser publicar um atalho de
linha de comando para o módulo.
