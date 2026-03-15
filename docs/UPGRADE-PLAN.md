# AI Smart Folders Upgrade Plan

## Status atual
O projeto saiu do estado de script unico e ja tem uma base nova, modular, convivendo com o fluxo legado.

Hoje existem dois caminhos principais no repositorio:
- legado: `smart-folders_v2.py` e variantes anteriores
- novo fluxo: `main.py` + pacote `ai_smart_folders/`

## O que ja foi entregue

### Fundacao arquitetural
- entrada nova via CLI em `main.py`
- CLI estruturada com `Typer` e fallback seguro para `argparse`
- modelos `Pydantic` para config, envelope, extracao, classificacao, metricas e benchmark
- pipeline modular por estagios em `ai_smart_folders/pipeline.py`
- separacao de responsabilidades em modulos (`config`, `extractors`, `llm`, `taxonomy`, `storage`, `evaluation`)

### Inteligencia
- separacao entre `understand` e `classify`
- prompts distintos para entendimento e classificacao
- suporte a estrategia local-only com modelos por papel:
  - `router_model`
  - `understanding_model`
  - `classification_model`
  - `fallback_model`
- heuristicas deterministicas antes do LLM
- suporte a `confidence`, `reason` e `needs_review`

### Organizacao
- estrutura alvo em 2 niveis:
  - `organized_dir/<category_l1>/<category_l2>/arquivo`
- pastas tecnicas:
  - `_NeedsReview`
  - `_Duplicates`
  - `_FailedExtraction`
- aliases de taxonomia por nivel
- alinhamento com categorias ja existentes no destino

### Persistencia e operacao
- SQLite com `run_id`, historico de eventos e suporte a `undo-last-run`
- cache versionado por hash + prompt/modelo
- logs separados para agente e API
- deteccao de duplicatas por hash
- `dry-run`
- `benchmark`
- `reindex-taxonomy`

### Avaliacao
- suporte a benchmark por dataset JSONL
- exemplo inicial em `evaluation/sample_dataset.jsonl`

## Estado real do projeto hoje

### O fluxo novo ja esta funcional para
- subir a CLI
- ler configuracao estruturada
- processar inbox em `run` ou `dry-run`
- organizar por taxonomia em 2 niveis
- mandar casos ambiguos para revisao
- registrar execucoes no banco
- reverter a ultima execucao
- rodar benchmark com dataset

### O fluxo novo ainda nao esta completo em
- cobertura de formatos adicionais como `XLSX/XLS` e `EML/MSG`
- OCR multipagina realmente robusto em todos os cenarios
- benchmark com dataset real versionado no repositorio
- relatorio mais rico de `dry-run` por arquivo
- uso de pools separados por tipo de carga alem do `ThreadPoolExecutor`
- dataset de avaliacao com ground truth real
- melhoria dos extractors para HTML/JSON/XML mais especializados

## Direcao tecnica validada
Seguimos com:
- `Pydantic`
- `Typer`
- arquitetura local e incremental
- taxonomia em 2 niveis
- regras + LLM
- benchmark local

Continuamos adiando:
- `Ray`
- clusterizacao
- orquestracao distribuida

Motivo:
- o projeto ainda ganha muito mais com consolidacao funcional do que com infraestrutura mais pesada
- o custo arquitetural do `Ray` continua nao se pagando neste momento

## Roadmap atual

### Fase 1: consolidacao da base modular
Status: concluida em boa parte

Ja entregue:
- CLI
- models
- pipeline
- storage
- taxonomia inicial
- benchmark basico

Pendencias pequenas:
- limpar detalhes de documentacao
- expandir exemplos de config
- melhorar mensagens operacionais

### Fase 2: cobertura e qualidade de extracao
Status: em andamento

Proximo foco:
- melhoria de OCR
- tratamento melhor de MIME e encodings
- extracao mais especializada por tipo

### Fase 3: qualidade da decisao
Status: parcialmente entregue

Ja existe:
- understanding separado
- classification separada
- thresholds de review
- duplicatas por hash

Proximo foco:
- trilha de decisao mais explicita
- normalization prompt mais robusto
- melhor uso da taxonomia existente
- relatorio de `dry-run` por documento

### Fase 4: mensurabilidade e evolucao segura
Status: iniciada

Ja existe:
- benchmark com dataset JSONL
- modelos e relatorios de benchmark

Proximo foco:
- dataset real de avaliacao
- metricas por etapa
- criterios de aceite por regressao

### Fase 5: performance por estagio
Status: ainda nao iniciada de verdade

Proximo foco:
- separar I/O, OCR e inferencia local por pools dedicados
- avaliar `ProcessPoolExecutor` onde fizer sentido
- manter `Ray` apenas como opcao futura se os gargalos reais justificarem

## Arquivos principais do novo fluxo
- `main.py`
- `ai_smart_folders/models.py`
- `ai_smart_folders/config.py`
- `ai_smart_folders/cli.py`
- `ai_smart_folders/pipeline.py`
- `ai_smart_folders/extractors.py`
- `ai_smart_folders/llm.py`
- `ai_smart_folders/taxonomy.py`
- `ai_smart_folders/storage.py`
- `ai_smart_folders/evaluation.py`

## Conclusao
O upgrade nao e mais apenas uma intencao: ele ja comecou e a nova espinha dorsal do projeto esta em uso. O foco agora nao deve ser reabrir a arquitetura, e sim consolidar o fluxo novo com:
- mais formatos
- melhor benchmark
- melhor `dry-run`
- melhor qualidade de classificacao
- mais observabilidade
