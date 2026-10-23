from langchain_classic.output_parsers.structured import (
  ResponseSchema,
  StructuredOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from itera_llm.model import criar_modelo

feedback_produto = """
O Galaxy S24 é um smartphone premium compacto, com tela AMOLED de 6,2 polegadas, ótimo desempenho e um conjunto versátil de câmeras. Ele oferece câmera principal de 50 MP, lente ultrawide e teleobjetiva com zoom óptico de 3x. A Samsung também promete sete gerações de atualizações do Android e sete anos de segurança. Samsung Newsroom Brasil
pontos positivos
- Tamanho compacto e confortável para usar com uma mão;
- Excelente tela, com boa visibilidade e taxa de atualização de 120 Hz;
- Desempenho rápido para aplicativos, jogos e multitarefa;
- Câmeras versáteis, com bons resultados em fotos e vídeos;
- Construção premium e resistência à água;
- Longo período de atualizações;
- Recursos de inteligência artificial do Galaxy AI.
pontos negativos
- Bateria de 4.000 mAh apenas razoável para usuários mais exigentes;
- Carregamento de 25 W mais lento que o de vários concorrentes;
- Pode aquecer durante jogos ou tarefas pesadas;
- Câmeras representam uma evolução pequena em relação ao Galaxy S23;
- Tela compacta pode não agradar quem assiste muitos vídeos ou prefere jogar em telas maiores.
Para quem é indicado
É indicado para quem procura um celular Android pequeno, potente, durável e com boas câmeras, sem precisar das dimensões ou dos recursos avançados do modelo Ultra.
Conclusão
O Galaxy S24 ainda vale a pena, especialmente quando encontrado com um bom desconto. É uma ótima escolha para quem prioriza tamanho compacto, desempenho e suporte prolongado. Entretanto, quem busca bateria de longa duração, carregamento rápido ou tela grande deve considerar outros modelos. As câmeras são boas, mas não são o principal motivo para trocar um Galaxy S23 por ele.
"""

def prompt_template(schema_formatado: str) -> ChatPromptTemplate:
  review_template2 = ChatPromptTemplate.from_template("""
para o texto a seguir extraia as seguinte informações: 
produto: Nome do produto mencionado no texto.
pontos positivos: listando os pontos positivos.
pontos negativos: listando os pontos negativos.
recomendação: O cliente recomenda este produto? Responda True para sim e False para não.

Texto: {review}

{schema}
""", partial_variables={"schema": schema_formatado}
)

  return review_template2


def schema()->StructuredOutputParser:
  schema_produto = ResponseSchema(
   name="produto",
   type="string",
   description=" nome do produto mencionado no texto."
 )
  schema_negativo = ResponseSchema(
   name="pontos_negativos",
   type="list",
   description=" pontos negativo mencionado no texto."
 )
  schema_positivo = ResponseSchema(
   name="pontos_positivos",
   type="list",
   description="pontos positivo mencionado no texto."
 )
  schema_recomendacao = ResponseSchema(
   name="recomendacao",
   type="boolean",
   description="O cliente recomenda o produto? Responda true para sim ou false para não."
 )

  response_schema = [schema_negativo, schema_positivo, schema_produto, schema_recomendacao]
  output_parse = StructuredOutputParser.from_response_schemas(response_schema)
  return output_parse

def template_recomenda(llm: ChatOpenAI, review_template: ChatPromptTemplate, output_parse: StructuredOutputParser):
  mensagem = review_template.format_messages(review=feedback_produto)
  resposta = llm.invoke(mensagem)
  resposta_json = output_parse.parse(resposta.content)

  return resposta_json


def main() -> None:

  llm = criar_modelo()

  parser = schema()
  schema_formatado = parser.get_format_instructions()

  review_template = prompt_template(schema_formatado)

  resultado = template_recomenda(
    llm, review_template, parser
  )

 
  print(f"Produto: {resultado['produto']}")
  print(f"Recomendado: {'Sim' if resultado['recomendacao'] else 'Não'}")

  print("\nPontos positivos:")
  for ponto in resultado["pontos_positivos"]:
      print(f"- {ponto}")

  print("\nPontos negativos:")
  for ponto in resultado["pontos_negativos"]:
      print(f"- {ponto}")



if __name__ == "__main__":
  main()
