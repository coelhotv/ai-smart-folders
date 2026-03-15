# AI Smart Folders: Plano de Evolucao v3

## Resumo
Este plano evolui o AI Smart Folders de um script monolitico para uma plataforma local mais inteligente, eficiente, organizada e auditavel, sem aumentar a complexidade cedo demais.

A direcao escolhida incorpora:
- `Pydantic` para contratos internos e configuracao
- `Typer` para uma CLI moderna
- pipeline modular por estagios
- taxonomia semantica em 2 niveis
- melhor estrategia de prompts e modelos locais
- extractors mais robustos
- performance via pools locais e subprocessos

A direcao explicitamente adiada:
- `Ray` como runtime principal agora

Motivo:
- o projeto atual ainda e pequeno e local
- o maior ganho de valor esta em modularizacao, qualidade de extracao, taxonomia, prompts e avaliacao
- `Ray` agrega custo arquitetural alto cedo demais
- podemos manter a arquitetura pronta para migrar para `Ray` depois, se os gargalos reais justificarem

## Objetivo
Transformar a app em um organizador local capaz de:
- entender melhor o conteudo dos arquivos
- classificar com mais consistencia
- suportar mais formatos e extrair melhor seus dados
- organizar os arquivos em uma hierarquia semantica util
- operar com mais throughput e seguranca
- evoluir de forma medivel, com benchmark e avaliacao local

Este plano foi desenhado para ser implementado em fatias pequenas, para continuar viavel mesmo com o plano gratuito do ChatGPT.

## Estado Atual
Hoje a app ja oferece:
- execucao local em Python
- extracao para alguns formatos
- OCR opcional
- classificacao via Ollama
- cache por hash
- persistencia SQLite
- paralelismo simples com threads
- movimentacao dos arquivos para uma pasta por categoria

Limites atuais observados no codigo:
- pipeline monolitico e acoplado
- um unico prompt resolve entendimento e classificacao
- um unico modelo faz papeis diferentes
- estrutura de diretorios rasa
- extractors ainda limitados
- paralelismo sem separacao por tipo de carga
- ausencia de benchmark local
- ausencia de fluxo explicito de revisao humana

## Decisoes Arquiteturais

### Adotar agora
- `Pydantic` para contratos de dados e configuracao
- `Typer` para CLI
- pipeline por estagios
- `DocumentEnvelope` como contrato central
- regras deterministicas antes do LLM
- prompts separados por responsabilidade
- taxonomia em 2 niveis
- `confidence`, `reason` e `needs_review`
- cache versionado por hash + modelo + prompt
- benchmark local e dataset de avaliacao
- `dry-run` e `undo-last-run`

### Adiar
- `Ray`
- atores distribuidos
- filas distribuidas
- infraestrutura de cluster
- orquestracao pesada

### Justificativa para nao adotar Ray agora
As recomendacoes do `upgrade_plan_v2.md` sobre `Ray` sao boas em abstrato, mas o custo de entrada nao se paga ainda para este projeto.

Riscos de introduzir `Ray` cedo:
- aumenta muito a complexidade da Etapa 1
- encarece debug, observabilidade e manutencao
- exige refatoracao estrutural pesada antes de entregar ganhos de produto
- pode consumir mais interacoes e tokens do que o necessario no ChatGPT

Conclusao:
- `Pydantic` e `Typer` entram agora
- `Ray` vira uma etapa opcional futura de escala

## Arquitetura Proposta

### Pipeline em estagios
A app passa a operar em um fluxo explicito:

1. `ingest`
2. `extract`
3. `understand`
4. `classify`
5. `act`
6. `review`

### Contrato central: `DocumentEnvelope`
Implementar em `Pydantic`.

Campos sugeridos:
- `document_id`
- `run_id`
- `source_path`
- `filename`
- `extension`
- `mime_type`
- `file_hash`
- `file_size`
- `metadata`
- `extracted_text`
- `extraction_quality`
- `ocr_used`
- `conversion_used`
- `summary`
- `keywords`
- `language`
- `document_type`
- `category_l1`
- `category_l2`
- `confidence`
- `reason`
- `needs_review`
- `destination_path`
- `status`
- `errors`

### Modelos adicionais
Tambem criar modelos `Pydantic` para:
- `AppConfig`
- `ExtractionResult`
- `ClassificationResult`
- `RunMetrics`
- `TaxonomyConfig`

### Estrutura sugerida de modulos
- `main.py`
- `cli/`
- `models/`
- `pipeline/`
- `extractors/`
- `classifiers/`
- `storage/`
- `taxonomy/`
- `evaluation/`

## Estrategia de Inteligencia

### Principio
Separar entendimento de classificacao.

Hoje a app usa um unico prompt para decidir categoria. O novo desenho deve ter etapas menores e mais especificas.

### Fluxo de inteligencia
1. extrair texto e metadados
2. gerar entendimento estruturado
3. classificar em taxonomia
4. normalizar nomes
5. revisar casos ambiguos

### Politica de modelos
A estrategia confirmada e `local-only`, mas nao `single-model`.

Papeis:
- `router_model`: leve e barato para triagem, idioma e tipo documental
- `understanding_model`: resumo e extracao semantica
- `classification_model`: decisao de pasta
- `fallback_model`: opcional, ainda local, para casos dificeis

### Regras deterministicas
Antes do LLM:
- regras de extensao e MIME
- regras por padroes fortes no nome do arquivo
- regras por metadados confiaveis
- regras para duplicatas

### Saida esperada do entendimento
- `summary`
- `keywords`
- `language`
- `document_type`
- `entities` opcionais
- `content_signals`

### Saida esperada da classificacao
- `category_l1`
- `category_l2`
- `confidence`
- `reason`
- `needs_review`

### Politica de baixa confianca
Se `confidence` ficar abaixo do threshold:
- nao inventar categoria
- mover para `_NeedsReview`
- persistir o motivo
- registrar sinais usados na decisao

## Estrategia de Prompts

### Prompt 1: Understanding
Responsabilidade:
- resumir o arquivo
- identificar idioma
- determinar tipo documental
- extrair temas e sinais relevantes

### Prompt 2: Classification
Responsabilidade:
- mapear o documento para `category_l1` e `category_l2`
- justificar em poucas palavras
- informar `confidence`
- marcar `needs_review` quando necessario

### Prompt 3: Normalization
Responsabilidade:
- converter nomes sugeridos em nomes canonicos
- alinhar com aliases existentes
- garantir seguranca para filesystem

### Regras de projeto para prompts
- responder sempre em JSON estruturado
- manter prompts curtos e especializados
- versionar prompts
- registrar qual prompt e modelo geraram cada decisao
- invalidar cache quando prompt ou modelo mudarem

## Estrategia de Extracao e Interpretacao

### Objetivo
Melhorar a qualidade do material de entrada do classificador. Hoje, boa parte da inteligencia da app depende diretamente da qualidade da extracao.

### Interface padrao de extractor
Cada extractor deve devolver:
- texto extraido
- metadados
- score de qualidade
- paginas, slides ou abas processadas
- flags de OCR e conversao
- erros e fallbacks usados

### Prioridade de cobertura
Primeira onda:
- PDF
- DOCX
- DOC
- PPTX
- PPT
- TXT
- MD
- CSV
- LOG
- JPG/JPEG/PNG

Segunda onda:
- XLSX/XLS
- EML/MSG
- HTML
- JSON/XML

Terceira onda opcional:
- ZIP como conteiner
- formatos adicionais especificos do usuario

### Melhorias obrigatorias
- MIME sniffing alem da extensao
- OCR multipagina real para PDFs escaneados
- chunking inteligente de documentos longos
- zonas prioritarias de leitura
- tratamento melhor de encoding
- fallback claro por tipo de arquivo

### Politica para documentos longos
Montar `content_preview` de forma mais inteligente:
- titulo e metadados
- inicio do documento
- headings principais
- trechos centrais relevantes
- conclusao ou final

## Organizacao de Diretorios

### Estrutura alvo
`organized_dir/<category_l1>/<category_l2>/arquivo.ext`

### Pastas tecnicas
- `_NeedsReview`
- `_Unsupported`
- `_Duplicates`
- `_FailedExtraction`

### Taxonomia
- `category_l1`: dominio semantico amplo
- `category_l2`: subtema, tipo documental ou projeto
- aliases e sinonimos devem convergir para nomes canonicos
- criacao de novas categorias deve ser controlada

### Regras
- nao usar o path original como verdade
- usar o path original apenas como sinal fraco
- evitar proliferacao de nomes quase iguais
- preservar nomes de arquivo com sanitizacao e deduplicacao

## Performance e Concorrencia

### Problema atual
Hoje cada thread executa o fluxo inteiro:
- leitura
- extracao
- LLM
- movimento
- persistencia

Isso mistura cargas I/O-bound, CPU-bound e inference-bound.

### Nova abordagem
Separar pools por tipo de trabalho:
- pool I/O para leitura, hash, scanning e move
- pool CPU/processo para OCR e conversoes pesadas
- executor limitado para inferencia local
- fila entre estagios para desacoplar gargalos

### Mecanismos sugeridos
- `ThreadPoolExecutor` para I/O
- `ProcessPoolExecutor` quando fizer sentido
- subprocessos isolados para OCR e LibreOffice
- limites conservadores para o modelo local

### Beneficio
- menos contencao
- melhor throughput
- mais previsibilidade
- caminho limpo para migrar para `Ray` se um dia for necessario

## Persistencia, Logs e Observabilidade

### Persistencia
Expandir o SQLite para registrar:
- `run_id`
- `document_id`
- prompt version
- modelo usado
- confidence
- destino sugerido
- destino efetivo
- flags de OCR/conversao
- motivo de revisao
- status por etapa

### Logs
Adicionar logs estruturados por:
- `run_id`
- `document_id`
- `stage`
- `duration_ms`

### Metricas
Medir:
- latencia por etapa
- taxa de cache hit
- taxa de OCR
- taxa de review
- taxa de erro
- confianca media
- distribuicao por categoria
- throughput por lote

## Funcionalidades de Produto

### `dry-run`
Permite:
- sugerir destino final
- mostrar motivo da classificacao
- nao mover nenhum arquivo

### `undo-last-run`
Permite:
- reverter a ultima execucao
- com base na persistencia do SQLite

### `benchmark`
Permite:
- comparar prompts
- comparar modelos locais
- medir acuracia, review rate e throughput

### `reindex-taxonomy`
Permite:
- reconstruir aliases
- recalcular catalogo de categorias
- validar consistencia da estrutura existente

## Proposta de Execucao em Etapas

### Etapa 1: Fundacao com Pydantic + Typer
Objetivo:
- criar a base modular sem reescrever toda a logica de negocio de uma vez

Entregas:
- criar nova estrutura de pastas do projeto
- introduzir `main.py` com `Typer`
- definir `AppConfig` e `DocumentEnvelope` em `Pydantic`
- encapsular a logica atual em estagios claros
- disponibilizar comando inicial `run`

Risco:
- baixo

Complexidade:
- media

Estimativa de tokens:
- `10k a 18k`

Como executar no plano gratuito:
- dividir em 2 a 3 conversas pequenas

### Etapa 2: Inteligencia separada por responsabilidade
Objetivo:
- melhorar acuracia sem aumentar demais o custo de inferencia

Entregas:
- separar understanding de classification
- introduzir `confidence`, `reason` e `needs_review`
- adicionar regras deterministicas
- suportar multiplos modelos locais por papel
- versionar prompts

Risco:
- medio

Complexidade:
- media a alta

Estimativa de tokens:
- `16k a 30k`

Como executar no plano gratuito:
- fazer primeiro understanding
- depois classification
- depois thresholds e review

### Etapa 3: Taxonomia e estrutura em 2 niveis
Objetivo:
- melhorar radicalmente a organizacao final entregue ao usuario

Entregas:
- migrar para `category_l1/category_l2`
- criar aliases e nomes canonicos
- definir politica de criacao de categorias
- adicionar `_NeedsReview`, `_Unsupported`, `_Duplicates`, `_FailedExtraction`

Risco:
- medio

Complexidade:
- media

Estimativa de tokens:
- `10k a 18k`

Como executar no plano gratuito:
- excelente etapa para uma conversa isolada

### Etapa 4: Extractors e qualidade de extracao
Objetivo:
- aumentar cobertura e qualidade dos dados que entram no pipeline

Entregas:
- criar extractor registry
- padronizar retorno dos extractors
- adicionar MIME sniffing
- melhorar OCR multipagina
- priorizar formatos faltantes

Risco:
- medio

Complexidade:
- alta

Estimativa de tokens:
- `14k a 28k`

Como executar no plano gratuito:
- quebrar por familia de formatos

### Etapa 5: Performance por estagio
Objetivo:
- melhorar throughput local com baixo risco

Entregas:
- filas locais entre estagios
- pools separados por carga
- subprocessos isolados para OCR e conversao
- controles de concorrencia por estagio

Risco:
- medio

Complexidade:
- media a alta

Estimativa de tokens:
- `14k a 26k`

Como executar no plano gratuito:
- primeiro separar LLM de OCR
- depois otimizar I/O

### Etapa 6: Benchmark, dry-run e undo
Objetivo:
- tornar a evolucao segura e mensuravel

Entregas:
- comando `dry-run`
- comando `benchmark`
- comando `undo-last-run`
- dataset de avaliacao local
- relatorio de qualidade e throughput

Risco:
- baixo

Complexidade:
- media

Estimativa de tokens:
- `12k a 22k`

Como executar no plano gratuito:
- benchmark e dry-run podem ser feitos em conversas separadas

### Etapa 7: Escala opcional com Ray
Objetivo:
- adotar runtime mais sofisticado apenas se houver necessidade comprovada

Entrar nesta etapa somente se houver:
- gargalo serio de OCR em lote
- necessidade de isolamento mais forte
- saturacao recorrente do host local
- limitacoes claras dos pools locais

Entregas:
- backend opcional com `Ray`
- atores por estagio
- manutencao do mesmo contrato `DocumentEnvelope`

Risco:
- medio a alto

Complexidade:
- alta

Estimativa de tokens:
- `18k a 34k`

Observacao:
- esta etapa nao e necessaria para o v1 evoluido entregar valor

## Estimativa Total de Tokens

### Faixa acumulada sem Ray
- Etapa 1: `10k a 18k`
- Etapa 2: `16k a 30k`
- Etapa 3: `10k a 18k`
- Etapa 4: `14k a 28k`
- Etapa 5: `14k a 26k`
- Etapa 6: `12k a 22k`

Total:
- minimo aproximado: `76k`
- maximo aproximado: `142k`

### Faixa acumulada com Ray depois
- adicionar Etapa 7: `18k a 34k`

Total expandido:
- minimo aproximado: `94k`
- maximo aproximado: `176k`

### Conclusao pratica sobre o plano gratuito
Sim, e viavel evoluir a app no plano gratuito, desde que a execucao seja feita em fatias pequenas.

A melhor estrategia e:
- uma subetapa por conversa
- escopo curto
- contexto pequeno
- validacao local ao final de cada fatia
- documentacao atualizada no repositorio para reduzir dependencia do historico do chat

Na pratica, o ideal e trabalhar com conversas de:
- `5k a 15k tokens` na maioria das implementacoes
- `10k a 20k` em refatoracoes medias
- evitar pedidos grandes como "implemente as etapas 1 a 4"

## Backlog Recomendado para Caber no Plano Gratuito

### Blocos ideais de conversa
1. criar `AppConfig` e `DocumentEnvelope`
2. adicionar `Typer` e comando `run`
3. separar pipeline em modulos
4. criar resultado estruturado de understanding
5. criar resultado estruturado de classification
6. adicionar `confidence` e `_NeedsReview`
7. introduzir taxonomia 2 niveis
8. criar aliases e normalizacao
9. criar extractor registry
10. melhorar PDF + OCR
11. melhorar DOCX/PPTX
12. adicionar XLSX/EML
13. separar pools por estagio
14. adicionar `dry-run`
15. adicionar benchmark
16. adicionar `undo-last-run`

### Diretriz operacional
Cada conversa deve:
- mexer em uma unica fatia
- incluir validacao ou teste
- atualizar a documentacao relevante
- evitar reabrir toda a arquitetura do projeto

## Criterios de Sucesso
Consideraremos a evolucao bem-sucedida quando:
- a classificacao estiver mais precisa e consistente
- a estrutura final estiver organizada em 2 niveis
- casos ambiguos forem corretamente enviados para `_NeedsReview`
- a taxa de falha de extracao cair
- o throughput melhorar em lotes maiores
- houver benchmark local para comparar prompts e modelos
- a app continuar funcional de forma local e confiavel

## Avaliacao das Recomendacoes do upgrade_plan_v2.md

### Recomendacoes aceitas
- usar `Pydantic` para contratos internos
- usar `Typer` para CLI moderna
- manter ideia de pipeline explicito por estagios
- preservar o conceito de evolucao orientada a performance por pools especializados

### Recomendacoes parcialmente aceitas
- conceito de separar responsabilidades como se fossem atores:
  aceito em nivel conceitual, mas implementado primeiro com modulos, filas locais e executores padrao

### Recomendacoes adiadas
- `Ray` como fundacao da nova arquitetura:
  adiado para etapa opcional de escala

### Motivo final
O `upgrade_plan_v2.md` melhora bem o desenho de medio prazo, mas para este momento a melhor decisao e manter o ganho de simplicidade e reduzir risco. O plano v3 absorve as partes mais valiosas dele sem transformar a fundacao em um projeto de infraestrutura antes da hora.

## Conclusao
A app ja tem um bom MVP. O maior salto de valor agora vira de:
- modularizar o pipeline
- melhorar a extracao
- separar entendimento de classificacao
- organizar melhor a taxonomia
- adicionar revisao, benchmark e observabilidade
- tratar performance por estagio

Esse plano preserva ambicao tecnica, mas com uma ordem de implementacao mais segura e mais compativel com o plano gratuito do ChatGPT.
