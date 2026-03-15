# AI Smart Folders

AI Smart Folders e uma app local em Python para organizar arquivos com ajuda de LLMs e heuristicas, transformando um inbox em uma estrutura semantica previsivel.

Hoje o projeto esta em transicao de um script monolitico para uma base modular mais robusta. O fluxo legado continua no repositorio, mas a nova espinha dorsal da app ja existe e esta sendo usada pela nova CLI.

## Estado atual

### O que a app ja faz
- processa arquivos locais a partir de um inbox
- extrai texto e metadados de varios formatos
- usa entendimento + classificacao separados
- organiza em uma taxonomia de 2 niveis
- envia casos ambiguos para revisao
- detecta duplicatas por hash
- persiste runs e movimentos em SQLite
- suporta `dry-run`, `undo-last-run`, `benchmark` e `reindex-taxonomy`

### O que estamos construindo agora
- ampliar cobertura de formatos e qualidade dos extractors
- enriquecer o `dry-run` com trilha de decisao por arquivo
- consolidar benchmark com dataset real
- adicionar metricas por etapa
- preparar a base para otimizar concorrencia por tipo de carga

## Arquitetura atual

### Fluxo legado
- `smart-folders_v2.py`
- `smart-folders_v1.py`

### Fluxo novo
- `main.py`
- `ai_smart_folders/`

Principais modulos do fluxo novo:
- `ai_smart_folders/models.py`
- `ai_smart_folders/config.py`
- `ai_smart_folders/cli.py`
- `ai_smart_folders/pipeline.py`
- `ai_smart_folders/extractors.py`
- `ai_smart_folders/llm.py`
- `ai_smart_folders/taxonomy.py`
- `ai_smart_folders/storage.py`
- `ai_smart_folders/evaluation.py`

## Funcionalidades implementadas no fluxo novo
- CLI com `Typer`
- modelos `Pydantic`
- pipeline por estagios:
  - `ingest`
  - `extract`
  - `understand`
  - `classify`
  - `act`
- prompts separados para entendimento e classificacao
- regras deterministicas antes do LLM
- taxonomia em 2 niveis com aliases
- pastas tecnicas:
  - `_NeedsReview`
  - `_Duplicates`
  - `_FailedExtraction`
- cache versionado por hash + modelo + prompt
- benchmark com dataset JSONL

## Formatos atualmente suportados no fluxo novo
- PDF
- DOCX
- DOC
- PPTX
- PPT
- TXT
- MD
- CSV
- LOG
- JSON
- XML
- HTML
- JPG
- JPEG
- PNG

## Requisitos
- Python 3.10 ou superior
- dependencias do `requirements.txt`
- `tesseract` para OCR quando usado
- `soffice` para conversao de `.doc` e `.ppt` quando necessario

Instalacao:

```bash
python -m pip install -r requirements.txt
```

## Configuracao
Use `config.example.yaml` como base.

O arquivo aceita configuracoes para:
- inbox e destino
- modelos locais
- thresholds de review
- workers
- taxonomia e aliases
- `data_dir`

Se voce mantiver `config.yaml` na raiz do projeto e nao definir `data_dir`, a nova CLI usa `./.ai-smart-folders-data/` para logs, cache e SQLite.

## Uso

### Nova CLI
```bash
python main.py run
python main.py dry-run
python main.py benchmark --limit 10
python main.py benchmark --dataset evaluation/sample_dataset.jsonl
python main.py undo-last-run
python main.py reindex-taxonomy
```

### Script legado
```bash
python smart-folders_v2.py
```

## Roadmap imediato

### Em andamento
- consolidar o fluxo novo como caminho principal
- melhorar extractors e cobertura de formatos
- tornar o benchmark util para regressao real

### Proximos passos
- adicionar `XLSX/XLS`
- adicionar `EML/MSG`
- melhorar OCR multipagina
- detalhar a saida de `dry-run`
- adicionar metricas por etapa
- avaliar pools dedicados para I/O, OCR e inferencia

### Deliberadamente adiado
- `Ray`
- infraestrutura distribuida
- clusterizacao

## Documentacao de trabalho
- [Upgrade Plan](docs/UPGRADE-PLAN.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Analise de Evolucao](docs/ANALISE_EVOLUCAO.md)
- [Analise de Stack](docs/ANALISE_STACK.md)

## Licenca
MIT. Veja [LICENSE](LICENSE).
