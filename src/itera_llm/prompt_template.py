from langchain_classic.output_parsers.structured import (
  ResponseSchema,
  StructuredOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from itera_llm.model import criar_modelo

feedback_produto = """"
O Galaxy S24 é um smartphone premium compacto, com tela AMOLED de 6,2 polegadas, ótimo desempenho e um conjunto versátil de câmeras. Ele oferece câmera principal de 50 MP, lente ultrawide e teleobjetiva com zoom óptico de 3x. A Samsung também promete sete gerações de atualizações do Android e sete anos de segurança. Samsung Newsroom Brasil
Pontos positivos
- Tamanho compacto e confortável para usar com uma mão;
- Excelente tela, com boa visibilidade e taxa de atualização de 120 Hz;
- Desempenho rápido para aplicativos, jogos e multitarefa;
- Câmeras versáteis, com bons resultados em fotos e vídeos;
- Construção premium e resistência à água;
- Longo período de atualizações;
- Recursos de inteligência artificial do Galaxy AI.
Pontos negativos
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

def prompt_template(llm: ChatOpenAI, schema_formatado: str) -> None:
  review_template = ChatPromptTemplate.from_template("""
para o texto a seguir extraia as seguinte informações: 
produto: Nome do produto mencionado no texto.
Pontos positivos: listando os pontos positivos.
Pontos negativos: listando os pontos negativos.
recoemdação: O cliente recomenda este produto? Responda True para sim e False para não.

Texto: {review}

retorne a resposta em JSON
""")
  review_template2 = ChatPromptTemplate.from_template("""
para o texto a seguir extraia as seguinte informações: 
produto: Nome do produto mencionado no texto.
Pontos positivos: listando os pontos positivos.
Pontos negativos: listando os pontos negativos.
recoemdação: O cliente recomenda este produto? Responda True para sim e False para não.

Texto: {review}

{schema}
""", partial_variables={"schema": schema_formatado}
)


def schema()->None:
  schema_produto = ResponseSchema(
   name="produto",
   type="string",
   description=" nome do produto mencionado no texto."
 )
  schema_negativo = ResponseSchema(
   name="Pontos negativo",
   type="list",
   description=" Pontos negativo mencionado no texto."
 )
  schema_positivo = ResponseSchema(
   name="Pontos positivo",
   type="list",
   description="Pontos positivo mencionado no texto."
 )
  schema_recomendacao = ResponseSchema(
   name="recomedação",
   type="bool",
   description="O cliente recomenda o produto? Responda True para sim ou False para não."
 )

  response_schema = [schema_negativo, schema_positivo, schema_produto, schema_recomendacao]
  output_parse = StructuredOutputParser.from_response_schemas(response_schema)
  schema_formatado = output_parse.get_format_instructions()
  print(schema_formatado)


def template_recomenda(llm: ChatOpenAI, review_template: str)-> None:
  mensagem = review_template.format_messages(review=feedback_produto)
  reposta = llm.invoke(mensagem)
  print(reposta.content)


def main() -> None:
  llm = criar_modelo()
  prompt_template(llm)
  schema()

if __name__ == "__main__":
  main()
