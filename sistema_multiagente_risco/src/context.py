from dataclasses import dataclass,field
from .schemas import DadosPaciente,ResultadoRisco
@dataclass
class PipelineContext:
    paciente:DadosPaciente|None=None; risco:ResultadoRisco|None=None; protocolos:list[str]=field(default_factory=list); etapas:list[str]=field(default_factory=list)
