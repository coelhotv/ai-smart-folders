# AI Smart Folders: Plano de Evolução v3

## Resumo
Evoluir o AI Smart Folders para uma plataforma local mais inteligente, eficiente, organizada e auditável, mantendo o foco em:
- execução local confiável
- evolução incremental
- baixo risco arquitetural
- implementação viável em conversas pequenas no ChatGPT

Incorporar do `upgrade_plan_v2.md`:
- `Pydantic` para contratos internos
- `Typer` para CLI moderna

Não incorporar agora como fundação:
- `Ray` como runtime principal

Em vez disso, adotar:
- pipeline por estágios
- filas locais
- `ThreadPoolExecutor` para I/O
- `ProcessPoolExecutor` ou subprocessos isolados para OCR/conversões pesadas
- limite explícito de concorrência para inferência local

## Decisões Arquiteturais
### Adotar agora
- `Pydantic` para `DocumentEnvelope`, config e payloads internos
- `Typer` para os comandos `run`, `dry-run`, `benchmark`, `undo-last-run`
- modularização explícita do pipeline em estágios
- taxonomia em 2 níveis
- confidence + `_NeedsReview`
- cache versionado por hash + prompt/modelo
- benchmark local e dataset de avaliação

### Adiar
- `Ray`
- filas distribuídas
- atores dedicados por estágio
- scheduler mais sofisticado que um pipeline local por processos/threads

### Justificativa
- o projeto atual é pequeno e local
- `Ray` adiciona runtime, acoplamento e custo cognitivo cedo demais
- `Pydantic` e `Typer` trazem benefício alto com risco baixo
- a maior parte do value prop vem antes de um framework de orquestração mais pesado

## Mudanças Principais
### 1. Fundação arquitetural
Reestruturar o código em módulos:
- `cli/`
- `models/`
- `pipeline/`
- `extractors/`
- `classifiers/`
- `storage/`
- `taxonomy/`
- `evaluation/`

Criar `DocumentEnvelope` em `Pydantic` com:
- identidade do documento
- metadados de origem
- conteúdo extraído
- qualidade da extração
- sinais de entendimento
- decisão de classificação
- destino final
- status e erros

Adicionar também modelos para:
- `AppConfig`
- `ExtractionResult`
- `ClassificationResult`
- `RunMetrics`

### 2. Inteligência e decisão
Separar o fluxo em:
- `understand`
- `classify`
- `normalize`

Saídas esperadas:
- `summary`
- `keywords`
- `language`
- `document_type`
- `category_l1`
- `category_l2`
- `confidence`
- `reason`
- `needs_review`

Usar estratégia local-only com múltiplos papéis:
- modelo leve para entendimento rápido
- modelo principal para classificação
- regras determinísticas antes do LLM
- fallback local mais forte opcional

### 3. Extração e interpretação
Implementar um registry de extractors com interface única.
Cobertura prioritária:
- PDF
- DOCX/DOC
- PPTX/PPT
- TXT/MD/CSV/LOG
- imagens
- depois XLSX/XLS e EML/MSG

Cada extractor deve devolver:
- texto
- metadados
- score de qualidade
- flags de OCR/conversão
- erro/fallback quando houver

### 4. Organização de diretórios
Migrar para:
- `organized_dir/<category_l1>/<category_l2>/arquivo`

Criar pastas técnicas:
- `_NeedsReview`
- `_Unsupported`
- `_Duplicates`
- `_FailedExtraction`

Taxonomia:
- `category_l1` como domínio amplo
- `category_l2` como subtema/tipo/projeto
- aliases para convergir sinônimos
- política de criação controlada de novas categorias

### 5. Performance local
Substituir o processamento “uma thread faz tudo” por estágios com concorrência específica:
- pool I/O para leitura, hash, descoberta, move
- pool CPU/processos para OCR e conversão
- executor limitado para chamadas ao modelo local

Usar filas locais entre estágios.
Se a carga real justificar no futuro, essa arquitetura pode ser migrada para `Ray` sem reescrever a lógica de domínio.

### 6. Produto, segurança e evolução
Adicionar:
- `dry-run`
- `undo-last-run`
- benchmark local
- dataset de avaliação
- logs estruturados por `run_id` e `document_id`
- métricas por etapa
- histórico de decisões e confiança

## Proposta de Execução em Etapas
### Etapa 1: Fundação com Pydantic + Typer
Entregas:
- nova estrutura modular
- `main` com `Typer`
- `DocumentEnvelope` e config em `Pydantic`
- pipeline básico sem mudar muito a lógica existente
- comando `run`

Esforço:
- implementação: média
- risco: baixo
- tokens estimados: `10k a 18k`

### Etapa 2: Separação da inteligência
Entregas:
- prompts separados de understanding e classification
- confidence/reason/needs_review
- regras determinísticas antes do LLM
- modelos configuráveis

Esforço:
- implementação: média/alta
- risco: médio
- tokens estimados: `16k a 30k`

### Etapa 3: Taxonomia e estrutura em 2 níveis
Entregas:
- `category_l1/category_l2`
- aliases
- criação controlada de categorias
- `_NeedsReview` e demais pastas técnicas

Esforço:
- implementação: média
- risco: médio
- tokens estimados: `10k a 18k`

### Etapa 4: Extractors e qualidade de extração
Entregas:
- extractor registry
- MIME sniffing
- score de qualidade
- OCR multipágina
- novos formatos prioritários

Esforço:
- implementação: alta
- risco: médio
- tokens estimados: `14k a 28k`

### Etapa 5: Performance por estágio
Entregas:
- filas locais
- pools separados por tipo de trabalho
- `ProcessPoolExecutor` ou subprocessos isolados para OCR/conversão
- limites explícitos para inferência local

Esforço:
- implementação: média/alta
- risco: médio
- tokens estimados: `14k a 26k`

### Etapa 6: Benchmark, dry-run e undo
Entregas:
- comando `dry-run`
- comando `benchmark`
- comando `undo-last-run`
- dataset de avaliação
- relatórios de qualidade e throughput

Esforço:
- implementação: média
- risco: baixo
- tokens estimados: `12k a 22k`

### Etapa 7: Escala opcional com Ray
Executar apenas se houver evidência de gargalo real.
Critérios para considerar:
- OCR pesado em lotes grandes
- múltiplos modelos locais concorrendo
- necessidade de isolamento mais forte por estágio
- saturação recorrente que os pools locais não resolvam

Entregas:
- troca do backend de orquestração por atores/filas `Ray`
- manter `DocumentEnvelope`, CLI e lógica de domínio

Esforço:
- implementação: alta
- risco: médio/alto
- tokens estimados: `18k a 34k`

## Interfaces Públicas
Adicionar comandos CLI:
- `run`
- `dry-run`
- `benchmark`
- `undo-last-run`
- `reindex-taxonomy`

Adicionar config para:
- modelos por papel
- thresholds de confiança
- workers por estágio
- flags de features
- taxonomia e aliases

## Testes e Cenários
Cobrir:
- PDF textual
- PDF escaneado
- DOCX/PPTX
- imagem OCR
- CSV/XLSX
- arquivo duplicado
- baixa confiança indo para `_NeedsReview`
- falha de extractor
- colisão de nome
- cache invalidado por mudança de prompt/modelo
- execução com múltiplos workers sem corromper DB/cache/destino

## Assumptions
- manter estratégia `local-only`
- foco em `CLI local confiável`
- organização em `2 níveis semânticos`
- `Pydantic` e `Typer` entram já no v3
- `Ray` fica como etapa opcional de escala, não como fundação
- a melhor forma de caber no plano gratuito é executar uma subetapa por conversa
