# CardioIA — Sistema Multiagente de Análise de Risco Sintético

## 1. Visão geral

O CardioIA é um projeto acadêmico que integra um modelo de Machine Learning a uma arquitetura multiagente para analisar dados sintéticos, estimar uma probabilidade de risco, classificar o resultado e sugerir protocolos de resposta.

A solução demonstra a integração entre aprendizado de máquina, agentes de IA, ferramentas computacionais, validação de dados e compartilhamento de contexto.

**Aviso:** o projeto utiliza dados e protocolos sintéticos, sem validação clínica. Não deve ser utilizado para diagnóstico, prescrição ou tomada de decisões médicas reais.

## 2. Objetivos

- Treinar e avaliar modelos supervisionados de classificação.
- Selecionar um modelo para estimar a probabilidade de pico de risco.
- Integrar o modelo a uma arquitetura multiagente.
- Consultar protocolos sintéticos conforme os dados e a classificação obtida.
- Consolidar os resultados em uma resposta estruturada.

## 3. Arquitetura multiagente

A solução possui quatro agentes:

1. **Orquestrador:** coordena o fluxo de execução e o encaminhamento entre os agentes.
2. **Analista de Risco:** utiliza o modelo de Machine Learning para estimar e classificar o risco.
3. **Especialista em Protocolos:** consulta as regras e seleciona os protocolos sintéticos aplicáveis.
4. **Planejador de Resposta:** organiza os resultados e apresenta a resposta final.

A arquitetura utiliza handoffs entre agentes, function tools, contexto compartilhado, `SQLiteSession` para histórico de interações e Pydantic para validação e estruturação dos dados.

### Fluxo de execução

1. Recebimento e validação dos dados de entrada.
2. Registro das informações no contexto compartilhado.
3. Predição da probabilidade de risco pelo modelo treinado.
4. Classificação do resultado em baixo, moderado ou alto.
5. Consulta dos protocolos sintéticos aplicáveis.
6. Consolidação da justificativa, dos protocolos e do aviso de uso acadêmico.

## 4. Modelo de Machine Learning

O treinamento utiliza uma base sintética com 5.000 registros e compara os modelos Regressão Logística e Random Forest.

O processo inclui preparação dos dados, separação entre treino e teste, treinamento, avaliação por métricas de classificação, análise de matrizes de confusão e seleção do modelo final.

Na execução integrada, o modelo utilizado foi um `RandomForestClassifier`.

### Resultados observados no conjunto de teste

| Métrica | Regressão Logística | Random Forest |
|---|---:|---:|
| Acurácia | 80,00% | 99,90% |
| Precisão da classe positiva | 56,27% | 99,59% |
| Recall da classe positiva | 82,45% | 100,00% |
| F1-score da classe positiva | 66,89% | 99,80% |

Esses resultados correspondem à execução registrada no notebook. O desempenho elevado deve ser interpretado considerando que os dados e o alvo são sintéticos e construídos segundo regras definidas no próprio exercício. Os resultados não representam validação clínica nem comprovam desempenho equivalente em dados reais.

## 5. Tecnologias utilizadas

- Python 3.11
- Pandas
- Scikit-learn
- Joblib
- Pydantic
- Google ADK
- SQLite
- Pytest

## 6. Estrutura do projeto

```text
sistema_multiagente_risco/
├── data/
│   ├── novo_caso.json
│   ├── novo_caso_baixo.json
│   └── protocolos.json
├── models/
│   ├── colunas_modelo.joblib
│   └── modelo_risco_cardiaco.joblib
├── src/
│   ├── agents_config.py
│   ├── context.py
│   ├── main.py
│   ├── schemas.py
│   └── tools.py
├── tests/
├── .env.example
├── README.md
├── requirements.txt
└── teste_gemini.py
```

A estrutura acima representa os principais arquivos do projeto. Arquivos adicionais podem existir no ambiente local.

## 7. Instalação e configuração

É necessário ter Python 3.11 instalado.

Crie e ative o ambiente virtual no Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r sistema_multiagente_risco/requirements.txt
```

Configure as variáveis de ambiente exigidas pelo projeto no arquivo `.env`, seguindo o modelo de `.env.example`. Não compartilhe chaves de API nem publique o arquivo `.env` no repositório.

Se o arquivo `requirements.txt` estiver na raiz do projeto, utilize o caminho correspondente à sua localização real.

## 8. Execução

Execute os comandos a partir da raiz do projeto:

```powershell
python -m sistema_multiagente_risco.src.main
```

Para executar os testes automatizados:

```powershell
python -m pytest -q
```

O fluxo principal apresenta uma resposta estruturada com probabilidade, classe prevista, classificação de risco, protocolos sugeridos, justificativa e aviso de uso acadêmico.

## 9. Testes e limitações

Na execução registrada durante o desenvolvimento, os testes automatizados apresentaram **44 testes aprovados**.

A solução possui limitações importantes:

- Os dados utilizados são sintéticos.
- Os protocolos representam regras acadêmicas simuladas.
- Não houve validação clínica com dados reais.
- As métricas dependem da forma como os dados e os rótulos foram gerados.
- A solução não deve orientar condutas assistenciais ou decisões médicas reais.

Como melhorias futuras, recomenda-se avaliar a solução com dados adequados e autorizados, revisar as regras de classificação, ampliar a avaliação de robustez e documentar os critérios de segurança e governança.

## 10. Uso acadêmico

Projeto desenvolvido para fins educacionais, com foco em Machine Learning, sistemas multiagentes e integração de ferramentas de IA.

Felipe Almeida - @Febdev27
Vinícius Burchert Vilas Boas - @Vburchert
Renan Wesler Nunes - @W-Nunes

**Repositório COLAB:** (https://colab.research.google.com/drive/1-t-i--ZQPG5pwzmvLfXEViLhTV4X_vO2)

**Demonstração:** inserir o link do vídeo de apresentação, quando disponível.
