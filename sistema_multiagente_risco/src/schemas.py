
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Define e valida os dados de entrada do paciente.
class DadosPaciente(BaseModel):
    # Impede o recebimento de campos não definidos no schema.
    model_config=ConfigDict(extra="forbid")

    # Valida os valores mínimos e máximos dos dados de entrada.
    idade_media:float=Field(ge=18,le=120); carga_sistema:float=Field(ge=0,le=100); tempo_espera:float=Field(ge=0,le=1440)
    fc_media:float=Field(ge=20,le=250); spo2_media:float=Field(ge=0,le=100); pa_sistolica:float=Field(ge=30,le=300)
    temperatura:float=Field(ge=0,le=50); densidade_eventos:int=Field(ge=0); eventos_24h:int=Field(ge=0); disponibilidade_recursos:float=Field(ge=0,le=100)


# Define a estrutura do resultado produzido pelo modelo de risco.
class ResultadoRisco(BaseModel):
    # Restringe a probabilidade, a classe prevista e o nível de risco.
    probabilidade:float=Field(ge=0,le=1); classe_prevista:Literal[0,1]; classificacao:Literal["BAIXO","MODERADO","ALTO"]; modelo:str


# Define o formato da resposta final apresentada pelo sistema.
class RespostaFinal(BaseModel):
    # Valida os dados do risco e os protocolos sugeridos.
    probabilidade:float=Field(ge=0,le=1); classe_prevista:Literal[0,1]; classificacao:Literal["BAIXO","MODERADO","ALTO"]
    protocolos_sugeridos:list[str]; justificativa:str=Field(min_length=20,max_length=1200); aviso:str

    # Valida a lista de protocolos, removendo itens duplicados.
    @field_validator("protocolos_sugeridos")
    @classmethod
    def sem_duplicatas(cls,v): return list(dict.fromkeys(v))
