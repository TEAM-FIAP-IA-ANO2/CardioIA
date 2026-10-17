import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sistema_multiagente_risco.src import tools
from sistema_multiagente_risco.src.schemas import DadosPaciente, ResultadoRisco


BASE = Path(__file__).resolve().parents[1]


def carregar_protocolos_teste():
    caminho = BASE / "data" / "protocolos.json"
    return json.loads(caminho.read_text(encoding="utf-8"))


def executar_protocolos(alteracoes=None, classificacao="MODERADO"):
    dados = {
        "idade_media": 35,
        "carga_sistema": 50,
        "tempo_espera": 30,
        "fc_media": 80,
        "spo2_media": 98,
        "pa_sistolica": 120,
        "temperatura": 36.6,
        "densidade_eventos": 1,
        "eventos_24h": 0,
        "disponibilidade_recursos": 90,
    }
    dados.update(alteracoes or {})
    paciente = DadosPaciente.model_validate(dados)

    risco = ResultadoRisco(
        probabilidade=0.5 if classificacao == "MODERADO" else 0.8,
        classe_prevista=0 if classificacao == "MODERADO" else 1,
        classificacao=classificacao,
        modelo="ModeloTeste",
    )

    contexto = SimpleNamespace(
        state={
            "paciente": paciente.model_dump(),
            "risco": risco.model_dump(),
            "etapas": ["paciente_validado", "risco_calculado"],
        }
    )

    protocolos = tools.consultar_protocolos(contexto)
    return protocolos, contexto


@pytest.mark.parametrize(
    "campo, valores, indice, esperados",
    [
        ("spo2_media", [87.9, 88.0, 88.1], 0, [True, False, False]),
        ("carga_sistema", [84.9, 85.0, 85.1], 1, [False, False, True]),
        ("fc_media", [119.9, 120.0, 120.1], 2, [False, False, True]),
        ("eventos_24h", [4, 5, 6], 3, [False, False, True]),
        ("disponibilidade_recursos", [29.9, 30.0, 30.1], 4, [True, False, False]),
        ("tempo_espera", [89.9, 90.0, 90.1], 5, [False, False, True]),
        ("temperatura", [37.9, 38.0, 38.1], 6, [False, False, True]),
        ("densidade_eventos", [5, 6, 7], 7, [False, False, True]),
        ("pa_sistolica", [89.9, 90.0, 90.1], 8, [True, False, False]),
    ],
)
def test_limites_numericos(campo, valores, indice, esperados, monkeypatch):
    protocolos_json = carregar_protocolos_teste()
    assert len(protocolos_json) == 10
    monkeypatch.setattr(
        tools, "carregar_protocolos", lambda: protocolos_json
    )

    descricao = protocolos_json[indice]

    for valor, deve_acionar in zip(valores, esperados):
        protocolos, contexto = executar_protocolos({campo: valor})

        assert (descricao in protocolos) == deve_acionar, (
            f"Campo={campo}, valor={valor}, "
            f"esperado acionamento={deve_acionar}, protocolos={protocolos}"
        )
        assert "protocolos_consultados" in contexto.state["etapas"]


def test_protocolo_de_risco_alto(monkeypatch):
    protocolos_json = carregar_protocolos_teste()
    monkeypatch.setattr(
        tools, "carregar_protocolos", lambda: protocolos_json
    )

    descricao = protocolos_json[9]

    protocolos_moderado, _ = executar_protocolos(
        classificacao="MODERADO"
    )
    protocolos_alto, _ = executar_protocolos(
        classificacao="ALTO"
    )

    assert descricao not in protocolos_moderado
    assert descricao in protocolos_alto