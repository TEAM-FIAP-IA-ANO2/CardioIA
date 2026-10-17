
import json
from pathlib import Path

from sistema_multiagente_risco.src.schemas import DadosPaciente
from sistema_multiagente_risco.src.tools import classificar


BASE_DIR = Path(__file__).resolve().parents[1]


def test_faixas():
    assert [
        classificar(x) for x in (0.1, 0.5, 0.9)
    ] == ["BAIXO", "MODERADO", "ALTO"]


def test_schema():
    caminho_caso = BASE_DIR / "data" / "novo_caso.json"

    with caminho_caso.open("r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    paciente = DadosPaciente(**dados)

    assert paciente.spo2_media == 89
