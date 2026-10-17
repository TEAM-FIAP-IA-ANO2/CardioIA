

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from .agents_config import orquestrador
from .schemas import DadosPaciente, RespostaFinal


# Define a pasta do projeto e carrega as variáveis de ambiente.
BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE / ".env")


async def carregar_ou_criar_sessao(
    session_service,
    app_name: str,
    user_id: str,
    session_id: str,
):
    """Recupera uma sessão existente ou cria uma nova."""

    # Consulta o histórico para verificar se a sessão já existe.
    sessao = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # Cria a sessão caso ela ainda não exista.
    if sessao is None:
        sessao = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

    return sessao


async def executar(caso: DadosPaciente, session_id: str):
    """Executa o pipeline multiagente do CardioIA."""

    # Verifica se a chave de acesso ao Gemini foi configurada.
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY não encontrada. Confira o arquivo .env."
        )

    app_name = "cardioia"
    user_id = "usuario_local"

    # Configura o banco SQLite utilizado para persistir as sessões.
    caminho_db = (BASE / "historico_agentes.db").resolve()
    db_url = f"sqlite+aiosqlite:///{caminho_db.as_posix()}"

    session_service = DatabaseSessionService(db_url=db_url)

    # Recupera ou inicializa o histórico da execução.
    await carregar_ou_criar_sessao(
        session_service=session_service,
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # Associa o agente Orquestrador ao mecanismo de execução do ADK.
    runner = Runner(
        app_name=app_name,
        agent=orquestrador,
        session_service=session_service,
    )

    # Envia os dados validados ao sistema multiagente.
    mensagem = types.Content(
        role="user",
        parts=[
            types.Part(
                text=(
                    "Execute o pipeline completo do CardioIA "
                    "usando exclusivamente os dados deste caso. "
                    "Não invente valores.\n\n"
                    + caso.model_dump_json(indent=2)
                )
            )
        ],
    )

    resposta_texto = None

    # Processa os eventos gerados pelos agentes durante a execução.
    async for evento in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=mensagem,
    ):
        # Captura somente o evento que contém a resposta final.
        if evento.is_final_response() and evento.content:
            partes = evento.content.parts or []

            # Extrai os trechos de texto disponíveis na resposta.
            textos = [
                parte.text
                for parte in partes
                if getattr(parte, "text", None)
            ]

            if textos:
                resposta_texto = "\n".join(textos)

    # Interrompe a execução se os agentes não gerarem uma resposta.
    if not resposta_texto:
        raise RuntimeError(
            "O pipeline terminou sem produzir uma resposta final."
        )

    # Converte a resposta JSON e trata possíveis erros de formato.
    try:
        dados_resposta = json.loads(resposta_texto)
    except json.JSONDecodeError as erro:
        raise RuntimeError(
            "A resposta final não veio em formato JSON válido. "
            f"Conteúdo recebido: {resposta_texto}"
        ) from erro

    # Valida a estrutura final antes de devolver o resultado.
    return RespostaFinal.model_validate(dados_resposta)


async def main():
    # Configura os argumentos aceitos pela linha de comando.
    parser = argparse.ArgumentParser(
        description="Executa o pipeline multiagente CardioIA."
    )

    # Permite informar um arquivo JSON diferente para o teste.
    parser.add_argument(
        "--caso",
        type=Path,
        default=BASE / "data" / "novo_caso.json",
        help="Caminho para o arquivo JSON com os dados do caso.",
    )

    # Permite reutilizar uma sessão informando seu identificador.
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Identificador opcional da sessão. "
            "Se omitido, será gerado automaticamente."
        ),
    )

    args = parser.parse_args()

    # Confere se o arquivo de entrada existe.
    caminho_caso = args.caso.resolve()

    if not caminho_caso.is_file():
        raise FileNotFoundError(
            f"Arquivo do caso não encontrado: {caminho_caso}"
        )

    # Lê o JSON e valida os dados com o schema do paciente.
    dados_caso = json.loads(
        caminho_caso.read_text(encoding="utf-8")
    )

    caso = DadosPaciente.model_validate(dados_caso)

    # Cria uma sessão independente por execução, por padrão.
    session_id = args.session_id or f"exec-{uuid.uuid4().hex[:12]}"

    print(f"ID da execução: {session_id}")
    print("Executando pipeline multiagente CardioIA...\n")

    # Executa os agentes e aguarda a resposta estruturada.
    resposta = await executar(caso, session_id)

    # Exibe o resultado final em formato JSON legível.
    print("\nResultado final:")
    print(resposta.model_dump_json(indent=2))


if __name__ == "__main__":
    # Inicia o programa assíncrono pela linha de comando.
    asyncio.run(main())
