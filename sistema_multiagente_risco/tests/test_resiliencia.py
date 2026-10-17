import pytest
from pydantic import ValidationError

from sistema_multiagente_risco.src.schemas import (
    DadosPaciente,
    RespostaFinal,
)


@pytest.fixture
def dados_paciente_validos():
    return {
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


@pytest.fixture
def resposta_valida():
    return {
        "probabilidade": 0.5,
        "classe_prevista": 1,
        "classificacao": "MODERADO",
        "protocolos_sugeridos": ["Protocolo A"],
        "justificativa": (
            "Resultado de teste para validar a estrutura "
            "da resposta final do sistema."
        ),
        "aviso": "Resultado exclusivamente acadêmico.",
    }


def test_rejeita_idade_abaixo_do_minimo(dados_paciente_validos):
    dados_paciente_validos["idade_media"] = 17

    with pytest.raises(ValidationError):
        DadosPaciente.model_validate(dados_paciente_validos)


def test_rejeita_campo_nao_definido(dados_paciente_validos):
    dados_paciente_validos["campo_inesperado"] = 123

    with pytest.raises(ValidationError):
        DadosPaciente.model_validate(dados_paciente_validos)


def test_rejeita_campo_obrigatorio_ausente(dados_paciente_validos):
    del dados_paciente_validos["spo2_media"]

    with pytest.raises(ValidationError):
        DadosPaciente.model_validate(dados_paciente_validos)


def test_rejeita_probabilidade_acima_de_um(resposta_valida):
    resposta_valida["probabilidade"] = 1.5

    with pytest.raises(ValidationError):
        RespostaFinal.model_validate(resposta_valida)


def test_remove_protocolos_duplicados(resposta_valida):
    resposta_valida["protocolos_sugeridos"] = [
        "Protocolo A",
        "Protocolo B",
        "Protocolo A",
    ]

    resultado = RespostaFinal.model_validate(resposta_valida)

    assert resultado.protocolos_sugeridos == [
        "Protocolo A",
        "Protocolo B",
    ]


def test_rejeita_classificacao_invalida(resposta_valida):
    resposta_valida["classificacao"] = "CRITICO"

    with pytest.raises(ValidationError):
        RespostaFinal.model_validate(resposta_valida)
