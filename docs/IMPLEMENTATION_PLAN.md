# AI Smart Folders Implementation Plan

## Estado atual
Este plano agora serve como documento operacional do projeto e nao apenas como proposta.

## Resumo de status

### Concluido
- documento de upgrade consolidado em `docs/UPGRADE-PLAN.md`
- nova CLI em `main.py`
- base modular em `ai_smart_folders/`
- modelos `Pydantic`
- config estruturada
- pipeline por estagios
- `dry-run`
- `undo-last-run`
- `benchmark`
- `reindex-taxonomy`
- taxonomia em 2 niveis
- aliases de taxonomia
- deteccao de duplicatas por hash
- benchmark com dataset JSONL
- filas e workers basicos por estagio (`extract`, `llm`, `act`)

### Parcialmente concluido
- extractors:
  ja existem para PDF, DOCX, DOC, PPTX, PPT, TXT, MD, CSV, LOG, JSON, XML, HTML, XLSX, XLS, EML, MSG, JPG, JPEG e PNG
- prompts:
  ja existe separacao entre understanding e classification, mas ainda falta sofisticar a normalizacao e a trilha de decisao
- review flow:
  ja existe `needs_review`, threshold de confianca, `_NeedsReview` e relatorio por arquivo, mas a apresentacao ainda pode ficar mais rica

### Nao iniciado ou ainda superficial
- ZIP como conteiner
- dataset real de benchmark
- metricas detalhadas por etapa
- `ProcessPoolExecutor` para OCR/conversoes
- relatorio detalhado de `dry-run` por arquivo

## O que esta pronto para uso agora

### Comandos
```bash
python main.py run
python main.py dry-run
python main.py benchmark --limit 10
python main.py benchmark --dataset evaluation/sample_dataset.jsonl
python main.py undo-last-run
python main.py reindex-taxonomy
```

### Fluxo novo
O fluxo novo ja faz:
1. ingestao de arquivos do inbox
2. extracao por tipo
3. entendimento estruturado
4. classificacao semantica
5. normalizacao de destino
6. roteamento para pasta final ou tecnica
7. persistencia no SQLite

## Plano de execucao daqui para frente

### Etapa 1: consolidacao da camada de extracao
Status: proxima

Objetivo:
- aumentar cobertura de formatos
- melhorar a qualidade do texto de entrada

Escopo:
- revisar tratamento de HTML/JSON/XML
- reforcar MIME sniffing e fallbacks
- melhorar OCR multipagina quando dependencias estiverem disponiveis

Critério de aceite:
- extractors novos integrados ao registry
- retorno padrao respeitado
- sem regressao nos formatos ja suportados

### Etapa 2: relatorio de `dry-run` e trilha de decisao
Status: iniciada

Objetivo:
- tornar a operacao auditavel e confiavel antes de mover arquivos

Escopo:
- emitir por arquivo:
  - categoria sugerida
  - confidence
  - reason
  - destino final
  - origem da decisao (regra, cache, modelo, duplicate)
- melhorar a saida JSON dos comandos

Critério de aceite:
- `dry-run` permitir revisao manual clara sem inspecao do banco

### Etapa 3: benchmark real
Status: iniciada

Objetivo:
- transformar benchmark em ferramenta de regressao real

Escopo:
- criar dataset real pequeno e versionado
- definir expectativas minimas por arquivo
- medir matches por `category_l1`, `category_l2` e `needs_review`
- adicionar exemplos reais anonimizados ou sinteticos

Critério de aceite:
- benchmark indicar regressao de classificacao entre mudancas

### Etapa 4: observabilidade e metricas
Status: futura

Objetivo:
- medir o comportamento por etapa

Escopo:
- duracao por stage
- taxa de cache hit
- taxa de OCR
- taxa de review
- taxa de duplicate
- confianca media por lote

Critério de aceite:
- relatorio de run com indicadores minimos no final da execucao

### Etapa 5: performance por estagio
Status: iniciada

Objetivo:
- reduzir gargalos sem introduzir infraestrutura pesada cedo demais

Escopo:
- evoluir os workers por estagio ja existentes
- separar melhor I/O, OCR e inferencia
- avaliar `ProcessPoolExecutor`
- manter compatibilidade com a arquitetura atual

Critério de aceite:
- ganho real de throughput sem degradar estabilidade

## Ordem recomendada de implementacao
1. extractors faltantes
2. `dry-run` detalhado
3. dataset real de benchmark
4. metricas por etapa
5. evolucao dos pools dedicados

## Riscos atuais
- benchmark ainda sem dataset real no repositorio
- cobertura de formatos ainda incompleta
- OCR ainda depende de disponibilidade local de ferramentas
- a nova base ainda convive com scripts legados, o que pode gerar confusao de entrada

## Decisao de produto mantida
Continuamos com:
- local-only
- CLI local confiavel
- taxonomia em 2 niveis
- evolucao incremental

Continuamos adiando:
- `Ray`
- distribuicao
- complexidade de cluster

## Definicao de pronto para a proxima milestone
Consideraremos a proxima milestone atingida quando:
- `dry-run` trouxer relatorio util por arquivo
- benchmark puder ser rodado com dataset real pequeno
- README e docs refletirem esse estado sem divergencia
