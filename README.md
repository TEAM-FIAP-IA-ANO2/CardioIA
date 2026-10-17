# FIAP --- Graduação ON em Inteligência Artificial

```{=html}
<p align="center">
```
`<a href="https://www.fiap.com.br/">`{=html}
`<img src="https://www.fiap.com.br/wp-content/themes/fiap2016/images/logo-fi ap.svg" alt="FIAP" width="40%">`{=html}
`</a>`{=html}
```{=html}
</p>
```
# 🚀 FASE 6 --- CardioIA

## Sistema Multiagente de Análise de Risco Sintético

------------------------------------------------------------------------

## 👩🏻‍💻 Sobre esta fase

Esta fase reúne o desenvolvimento do projeto acadêmico **CardioIA**, que
integra um modelo de Machine Learning a uma arquitetura multiagente para
analisar dados sintéticos, estimar uma probabilidade de risco e
consolidar uma resposta estruturada.

O repositório organiza o notebook de treinamento, os artefatos do
modelo, o código do sistema multiagente, os testes automatizados e a
documentação técnica.

> **Aviso de uso acadêmico:** o projeto utiliza dados e protocolos
> sintéticos, sem validação clínica. Não deve ser utilizado para
> diagnóstico, prescrição, triagem ou tomada de decisões médicas reais.

------------------------------------------------------------------------

## 🎯 Objetivo da fase

Desenvolver um protótipo acadêmico que demonstre a integração entre
Machine Learning supervisionado e agentes de IA, incluindo:

-   Geração e preparação de uma base de dados sintética.
-   Treinamento e avaliação de modelos de classificação.
-   Seleção de um modelo para estimar a probabilidade de pico de risco.
-   Integração do modelo a uma arquitetura multiagente.
-   Consulta a protocolos sintéticos conforme a classificação obtida.
-   Consolidação dos resultados em uma resposta estruturada.

------------------------------------------------------------------------

## 📚 Conteúdos abordados

-   Machine Learning supervisionado e classificação binária.
-   Preparação de dados e engenharia de atributos.
-   Avaliação de modelos por métricas de classificação e matriz de
    confusão.
-   Arquitetura multiagente, orquestração e handoffs.
-   Function tools e contexto compartilhado.
-   Validação de dados e contratos com Pydantic.
-   Persistência de histórico com SQLite.
-   Testes automatizados e documentação técnica.
-   Ética, limitações e uso responsável de IA em contexto de saúde.

------------------------------------------------------------------------

## 🛠 Tecnologias utilizadas

-   Python 3.11
-   Pandas
-   Scikit-learn
-   Joblib
-   Pydantic
-   Google ADK
-   Gemini / Google GenAI
-   SQLite
-   Pytest
-   Google Colab

------------------------------------------------------------------------

## 📂 Projeto desenvolvido

### 📌 CardioIA --- Sistema Multiagente de Análise de Risco Sintético

**Descrição:**\
O sistema combina um classificador treinado com dados sintéticos e
quatro agentes: Orquestrador, Analista de Risco, Especialista em
Protocolos e Planejador de Resposta. O fluxo valida a entrada, executa a
predição, consulta regras acadêmicas simuladas e organiza a resposta
final.

**Principais componentes:**

1.  **Orquestrador:** coordena o fluxo e encaminha a tarefa ao agente
    responsável.
2.  **Analista de Risco:** valida os campos e aciona o modelo de Machine
    Learning.
3.  **Especialista em Protocolos:** consulta as regras sintéticas
    aplicáveis.
4.  **Planejador de Resposta:** consolida a saída estruturada.

A solução utiliza handoffs entre agentes, tools, contexto compartilhado,
`SQLiteSession` para histórico e Pydantic para validação e estruturação
dos dados.

### Modelo de Machine Learning

O notebook gera 5.000 registros sintéticos e compara Regressão Logística
e Random Forest. Na execução integrada, o modelo utilizado é um
`RandomForestClassifier`.

Resultados registrados no notebook:

  Métrica                         Regressão Logística   Random Forest
  ----------------------------- --------------------- ---------------
  Acurácia                                     80,00%          99,90%
  Precisão da classe positiva                  56,27%          99,59%
  Recall da classe positiva                    82,45%         100,00%
  F1-score da classe positiva                  66,89%          99,80%

Esses resultados pertencem ao experimento com dados sintéticos. Como o
alvo foi definido por regras do próprio exercício, as métricas podem ser
excessivamente otimistas e não demonstram desempenho clínico em dados
reais.

------------------------------------------------------------------------

## 📓 Notebook no Google Colab

[**Abrir o notebook de treinamento no Google
Colab**](https://colab.research.google.com/drive/1-t-i--ZQPG5pwzmvLfXEViLhTV4X_vO2)

O notebook contempla a geração da base sintética, o treinamento e a
avaliação dos modelos, a análise da matriz de confusão, a simulação de
previsão para um novo caso e a exportação dos artefatos do modelo.

------------------------------------------------------------------------

## 🗂 Estrutura do projeto

``` text
Cap 1 - Gestão de Crises com Sistema Preditivo em IA_v3/
├── artefatos_modelo/
│   ├── colunas_modelo.joblib
│   ├── dados_sinteticos.csv
│   ├── metricas_modelo.json
│   ├── modelo_risco_cardiaco.joblib
│   └── protocolos.json
├── docs/
│   ├── documento_arquitetura_cardioia_atualizado.pdf
│   └── relatorio_tecnico_cardioia_atualizado.pdf
├── sistema_multiagente_risco/
│   ├── data/
│   ├── models/
│   ├── src/
│   │   ├── agents_config.py
│   │   ├── context.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── tools.py
│   ├── tests/
│   ├── .env.example
│   └── requirements.txt
├── treinamento_modelo_risco_cardiaco.ipynb
└── README.md
```

A estrutura apresenta os principais arquivos esperados; arquivos
adicionais podem existir no repositório. O arquivo `.env` real, bancos
de dados locais, caches e ambientes virtuais não devem ser versionados.

------------------------------------------------------------------------

## ⚙️ Instalação e configuração

É recomendado utilizar Python 3.11.

No Windows PowerShell, a partir da raiz do projeto:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\sistema_multiagente_risco\requirements.txt
```

Configure as variáveis de ambiente no arquivo
`sistema_multiagente_risco/.env`, usando
`sistema_multiagente_risco/.env.example` como referência. A configuração
utiliza Google ADK / Gemini e pode exigir uma chave válida da API
Google.

**Não inclua a chave real no README, no notebook compartilhado
publicamente ou no controle de versão.** O arquivo `.env` deve
permanecer local e ignorado pelo Git.

------------------------------------------------------------------------

## ▶️ Execução

Execute a partir da raiz do projeto:

``` powershell
python -m sistema_multiagente_risco.src.main
```

Para executar os testes automatizados:

``` powershell
python -m pytest -q
```

Na execução registrada durante o desenvolvimento, foram aprovados **44
testes**. A quantidade de testes pode variar caso o código seja
alterado.

O fluxo principal produz uma resposta estruturada com classificação,
probabilidade, protocolos sugeridos, justificativa e aviso de uso
acadêmico.

------------------------------------------------------------------------

## 📄 Entregáveis acadêmicos

1.  **Notebook no Google Colab:** geração de dados, treinamento,
    avaliação e simulação de previsão.
2.  **Relatório técnico (máximo de 2 páginas):** decisões do modelo,
    métricas, matriz de confusão, limitações e melhorias.
3.  **Documento de arquitetura (máximo de 3 páginas):** diagrama/fluxo,
    agentes, handoffs, tools e exemplo de entrada e saída.
4.  **Código completo do sistema multiagente:** agentes, orquestração e
    integração com o modelo.
5.  **README.md:** instruções de execução, dependências, estrutura,
    integrantes e link da demonstração.

Os PDFs atualizados devem ser mantidos na pasta `docs/` com os nomes
indicados na estrutura, caso esses sejam os nomes usados no repositório.

------------------------------------------------------------------------

## 🧠 Competências desenvolvidas

-   Estruturação de problemas de IA.
-   Construção e avaliação de modelos de Machine Learning.
-   Integração de modelos em aplicações Python.
-   Arquitetura multiagente e uso de ferramentas.
-   Validação e organização de dados.
-   Testes automatizados e versionamento de código.
-   Documentação e comunicação técnica.
-   Análise crítica de limitações, ética e governança.

------------------------------------------------------------------------

## 👥 Integrantes

-   **Felipe Almeida** --- GitHub:
    [@Febdev27](https://github.com/Febdev27) --- **RM:** preencher
-   **Vinícius Burchert Vilas Boas** --- GitHub:
    [@Vburchert](https://github.com/Vburchert) --- **RM:** 565395
-   **Renan Wesler Nunes** --- GitHub:
    [@W-Nunes](https://github.com/W-Nunes) --- **RM:** 562839

## 🎥 Demonstração

**Vídeo de apresentação:** inserir o link do vídeo quando estiver
disponível.

------------------------------------------------------------------------

## ⚠️ Limitações e uso responsável

-   Os dados e protocolos são sintéticos.
-   Não houve validação clínica com dados reais.
-   As métricas dependem da construção dos dados e dos rótulos no
    experimento.
-   O protótipo não deve orientar diagnóstico, triagem, prescrição ou
    decisões médicas.
-   Uma aplicação real exigiria dados autorizados, validação
    independente, avaliação de robustez e vieses, governança,
    monitoramento e supervisão especializada.

------------------------------------------------------------------------

## 📋 Licença e atribuição

Este repositório é um projeto acadêmico. Caso utilize o template
institucional da FIAP, preserve os créditos e a licença indicados no
modelo original e confirme as condições de uso antes de redistribuir
materiais institucionais.
