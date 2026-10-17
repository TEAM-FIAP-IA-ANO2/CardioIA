
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from sistema_multiagente_risco.src.schemas import (
    DadosPaciente,
    ResultadoRisco,
    RespostaFinal,
)
from sistema_multiagente_risco.src import tools


def criar_paciente(**alteracoes):
    """Cria um paciente sintético válido para os testes."""
    dados = {
        "idade_media": 45,
        "carga_sistema": 50,
        "tempo_espera": 30,
        "fc_media": 80,
        "spo2_media": 98,
        "pa_sistolica": 120,
        "temperatura": 36.5,
        "densidade_eventos": 2,
        "eventos_24h": 1,
        "disponibilidade_recursos": 80,
    }

    dados.update(alteracoes)

    return DadosPaciente(**dados)


def criar_contexto(estado=None):
    """Simula o estado utilizado pelas ferramentas do ADK."""
    return SimpleNamespace(state=estado or {})


# ============================================================
# TESTES DE VALIDAÇÃO DOS DADOS
# ============================================================

def test_aceita_paciente_valido():
    paciente = criar_paciente()

    assert paciente.idade_media == 45


def test_rejeita_idade_fora_do_limite():
    with pytest.raises(ValidationError):
        criar_paciente(idade_media=17)


def test_rejeita_spo2_acima_de_100():
    with pytest.raises(ValidationError):
        criar_paciente(spo2_media=101)


# ============================================================
# TESTES DE CLASSIFICAÇÃO DE RISCO
# ============================================================

@pytest.mark.parametrize(
    "probabilidade, esperado",
    [
        (0.0, "BAIXO"),
        (0.2999, "BAIXO"),
        (0.30, "MODERADO"),
        (0.6999, "MODERADO"),
        (0.70, "ALTO"),
        (1.0, "ALTO"),
    ],
)
def test_classificar_risco(probabilidade, esperado):
    assert tools.classificar(probabilidade) == esperado


# ============================================================
# TESTE DE REGISTRO DO PACIENTE
# ============================================================

def test_registrar_paciente_salva_dados_e_etapa():
    contexto = criar_contexto()

    resultado = tools.registrar_paciente(
        idade_media=45,
        carga_sistema=50,
        tempo_espera=30,
        fc_media=80,
        spo2_media=98,
        pa_sistolica=120,
        temperatura=36.5,
        densidade_eventos=2,
        eventos_24h=1,
        disponibilidade_recursos=80,
        tool_context=contexto,
    )

    assert resultado["idade_media"] == 45
    assert contexto.state["paciente"]["idade_media"] == 45
    assert "paciente_validado" in contexto.state["etapas"]


# ============================================================
# TESTE DE PREVISÃO DO MODELO
# ============================================================

def test_prever_risco_registra_resultado(monkeypatch):
    contexto = criar_contexto(
        {
            "paciente": criar_paciente().model_dump(),
            "etapas": ["paciente_validado"],
        }
    )

    class ModeloFalso:
        named_steps = {}

        def predict_proba(self, entrada):
            assert len(entrada) == 1

            # Retorna uma matriz NumPy, como o modelo real.
            return np.array([[0.0267, 0.9733]])

    monkeypatch.setattr(
        tools,
        "carregar_modelo",
        lambda: ModeloFalso(),
    )

    monkeypatch.setattr(
        tools,
        "carregar_colunas",
        lambda: list(criar_paciente().model_dump().keys()),
    )

    resultado = tools.prever_risco(contexto)

    assert resultado["probabilidade"] == 0.9733
    assert resultado["classificacao"] == "ALTO"
    assert resultado["classe_prevista"] == 1
    assert "risco_calculado" in contexto.state["etapas"]


def test_prever_risco_sem_paciente_falha():
    contexto = criar_contexto()

    with pytest.raises(ValueError, match="Registre o paciente"):
        tools.prever_risco(contexto)


# ============================================================
# TESTES DOS PROTOCOLOS
# ============================================================

def test_consultar_protocolos_aplica_regras(monkeypatch):
    paciente = criar_paciente(
        carga_sistema=90,
        disponibilidade_recursos=20,
    )

    risco = ResultadoRisco(
        probabilidade=0.8,
        classe_prevista=1,
        classificacao="ALTO",
        modelo="ModeloFalso",
    )

    contexto = criar_contexto(
        {
            "paciente": paciente.model_dump(),
            "risco": risco.model_dump(),
            "etapas": [
                "paciente_validado",
                "risco_calculado",
            ],
        }
    )

    monkeypatch.setattr(
        tools,
        "carregar_protocolos",
        lambda: [
            "Se carga do sistema > 85% → abrir unidades temporárias",
            "Se disponibilidade_recursos < 30% → redistribuir insumos",
            "Se risco sistêmico alto → ativar estado de emergência",
        ],
    )

    protocolos = tools.consultar_protocolos(contexto)

    assert len(protocolos) == 3
    assert protocolos == contexto.state["protocolos"]
    assert "protocolos_consultados" in contexto.state["etapas"]


def test_protocolos_nao_sao_gerados_sem_risco():
    contexto = criar_contexto(
        {
            "paciente": criar_paciente().model_dump(),
            "etapas": ["paciente_validado"],
        }
    )

    with pytest.raises(ValueError, match="Pipeline incompleto"):
        tools.consultar_protocolos(contexto)


# ============================================================
# TESTES DO SCHEMA DA RESPOSTA FINAL
# ============================================================

def test_resposta_final_valida():
    resposta = RespostaFinal(
        probabilidade=0.8,
        classe_prevista=1,
        classificacao="ALTO",
        protocolos_sugeridos=["Protocolo simulado"],
        justificativa=(
            "Resultado demonstrativo obtido com dados sintéticos "
            "para avaliação acadêmica do pipeline."
        ),
        aviso=(
            "Sistema acadêmico sem validação clínica. "
            "Não utilizar para decisões médicas reais."
        ),
    )

    assert resposta.classificacao == "ALTO"
    assert len(resposta.protocolos_sugeridos) == 1


def test_resposta_final_remove_protocolos_duplicados():
    resposta = RespostaFinal(
        probabilidade=0.8,
        classe_prevista=1,
        classificacao="ALTO",
        protocolos_sugeridos=[
            "Protocolo A",
            "Protocolo A",
        ],
        justificativa=(
            "Resultado demonstrativo obtido com dados sintéticos "
            "para avaliação acadêmica do pipeline."
        ),
        aviso="Sistema acadêmico sem validação clínica.",
    )

    assert resposta.protocolos_sugeridos == ["Protocolo A"]


@pytest.mark.parametrize(
    "alteracoes, probabilidade, classificacao, descricao",
    [
        (
            {"spo2_media": 87},
            0.5,
            "MODERADO",
            "Se SPO2 < 88% â†’ iniciar suporte respiratÃ³rio imediato",
        ),
        (
            {"carga_sistema": 86},
            0.5,
            "MODERADO",
            "Se carga do sistema > 85% â†’ abrir unidades temporÃ¡rias",
        ),
        (
            {"fc_media": 121},
            0.5,
            "MODERADO",
            "Se FC > 120 â†’ iniciar monitoramento intensivo",
        ),
        (
            {"eventos_24h": 6},
            0.5,
            "MODERADO",
            "Se eventos_24h > 5 â†’ ativar equipe de resposta rÃ¡pida",
        ),
        (
            {"disponibilidade_recursos": 29},
            0.5,
            "MODERADO",
            "Se disponibilidade_recursos < 30% â†’ redistribuir insumos",
        ),
        (
            {"tempo_espera": 91},
            0.5,
            "MODERADO",
            "Se tempo_espera > 90 min â†’ priorizar triagem emergencial",
        ),
        (
            {"temperatura": 38.1},
            0.5,
            "MODERADO",
            "Se temperatura > 38Â°C mÃ©dia â†’ iniciar protocolo infeccioso",
        ),
        (
            {"densidade_eventos": 7},
            0.5,
            "MODERADO",
            "Se densidade_eventos > 6 â†’ ativar plano de contingÃªncia",
        ),
        (
            {"pa_sistolica": 89},
            0.5,
            "MODERADO",
            "Se PA sistÃ³lica < 90 â†’ iniciar protocolo de choque",
        ),
        (
            {},
            0.8,
            "ALTO",
            "Se risco sistÃªmico alto â†’ ativar estado de emergÃªncia",
        ),
    ],
)
def test_cada_protocolo_e_acionado(
    monkeypatch,
    alteracoes,
    probabilidade,
    classificacao,
    descricao,
):
    paciente = criar_paciente(**alteracoes)

    risco = ResultadoRisco(
        probabilidade=probabilidade,
        classe_prevista=int(probabilidade >= 0.5),
        classificacao=classificacao,
        modelo="ModeloFalso",
    )

    contexto = criar_contexto(
        {
            "paciente": paciente.model_dump(),
            "risco": risco.model_dump(),
            "etapas": ["paciente_validado", "risco_calculado"],
        }
    )

    monkeypatch.setattr(
        tools,
        "carregar_protocolos",
        lambda: [descricao],
    )

    protocolos = tools.consultar_protocolos(contexto)

    assert protocolos == [descricao]
    assert contexto.state["protocolos"] == [descricao]
    assert "protocolos_consultados" in contexto.state["etapas"]
