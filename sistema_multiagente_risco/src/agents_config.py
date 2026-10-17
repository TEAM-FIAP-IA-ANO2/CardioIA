
import os

from google.adk.agents import LlmAgent, SequentialAgent

from .schemas import RespostaFinal
from .tools import (
    registrar_paciente,
    prever_risco,
    consultar_protocolos,
    obter_estado_pipeline,
)

# Define o modelo Gemini utilizado pelos agentes.
# Se GEMINI_MODEL não estiver configurado, utiliza o modelo padrão.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# Agente responsável por registrar os dados do paciente
# e executar o modelo preditivo de risco.
analista_risco = LlmAgent(
    name="AnalistaRisco",
    model=MODEL,
    description="Valida os dados e executa o modelo preditivo.",
    instruction="""
    Você é o Analista de Risco do projeto acadêmico CardioIA.

    Leia os dados JSON do caso enviados pelo usuário.
    Chame registrar_paciente usando os dez campos recebidos.
    Em seguida, chame prever_risco exatamente uma vez.
    Não invente dados nem calcule probabilidades manualmente.
    Ao terminar, informe que a análise foi registrada.
    """,
    # Disponibiliza as ferramentas de registro e previsão de risco.
    tools=[registrar_paciente, prever_risco],
)


# Agente responsável por consultar os protocolos
# simulados de acordo com os dados registrados na sessão.
especialista_protocolos = LlmAgent(
    name="EspecialistaProtocolos",
    model=MODEL,
    description="Consulta as regras de protocolos simulados.",
    instruction="""
    Você é o Especialista em Protocolos.

    Consulte os dados de risco e paciente já registrados na sessão.
    Chame consultar_protocolos exatamente uma vez.
    Não crie protocolos nem altere as regras retornadas.
    Ao terminar, informe que a consulta foi concluída.
    """,
    # Disponibiliza a ferramenta de consulta de protocolos.
    tools=[consultar_protocolos],
)


# Agente responsável por consolidar os resultados
# e gerar a resposta final no formato definido pelo schema.
planejador_resposta = LlmAgent(
    name="PlanejadorResposta",
    model=MODEL,
    description="Consolida e valida o resultado final.",
    instruction="""
    Você é o Planejador de Resposta do CardioIA.

    Chame obter_estado_pipeline exatamente uma vez.
    Utilize os valores retornados sem alterar a probabilidade,
    a classe prevista, a classificação ou os protocolos.

    Gere a resposta final conforme o schema RespostaFinal.
    A justificativa deve explicar que o resultado deriva de um
    modelo treinado com dados sintéticos.
    Inclua um aviso explícito de que o sistema é acadêmico,
    não tem validação clínica e não deve orientar decisões médicas reais.
    Não invente informações ausentes.
    """,
    # Disponibiliza a ferramenta que recupera o estado do pipeline.
    tools=[obter_estado_pipeline],
    # Exige que a resposta final siga a estrutura RespostaFinal.
    output_schema=RespostaFinal,
)


# Orquestrador que executa os três agentes sequencialmente,
# respeitando a ordem de análise, consulta e consolidação.
orquestrador = SequentialAgent(
    name="OrquestradorCardioIA",
    description="Executa o pipeline CardioIA em ordem controlada.",
    sub_agents=[
        analista_risco,
        especialista_protocolos,
        planejador_resposta,
    ],
)
