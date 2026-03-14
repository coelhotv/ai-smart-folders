# Documento a criar
Salvar como: `IMPLEMENTATION_PLAN.md`

# AI Smart Folders: Plano de Evolução Completo

## Objetivo
Evoluir o AI Smart Folders de um organizador baseado em um único prompt e um único modelo para uma plataforma local mais inteligente, eficiente, organizada, performática e auditável, capaz de processar múltiplos tipos de arquivos, classificar com maior precisão e manter uma estrutura consistente de diretórios.

Este plano foi desenhado com um objetivo prático adicional: **ser executável mesmo com o plano gratuito do ChatGPT**, dividindo a evolução em etapas pequenas, autocontidas e com consumo de tokens previsível.

---

## Estado atual resumido
Hoje a app já possui uma base útil:
- pipeline local em Python
- extração de texto para alguns formatos
- OCR opcional
- classificação via Ollama
- cache por hash
- persistência SQLite
- execução paralela simples com threads
- movimentação de arquivos para uma pasta por categoria

Limitações principais atuais:
- inteligência concentrada em um único prompt de classificação
- um único modelo para tarefas diferentes
- baixa separação entre extração, entendimento e decisão
- organização rasa de diretórios
- suporte de formatos ainda limitado
- paralelismo sem orquestração por estágio
- ausência de benchmark/eval para medir qualidade real
- ausência de fluxo de revisão para baixa confiança

---

## Visão alvo
Queremos chegar em uma app com estas propriedades:

### 1. Inteligência melhor
- separar entendimento, classificação e normalização em etapas distintas
- usar prompts menores e mais específicos
- usar estratégia local-only, mas com mais de um modelo configurável
- combinar regras determinísticas + LLM
- decidir com confiança e encaminhar casos ambíguos para revisão

### 2. Eficiência melhor
- interpretar mais formatos e extrair melhor conteúdo
- melhorar OCR, MIME detection, chunking e metadados
- evitar reprocessamento com cache versionado
- aproveitar melhor o contexto útil de documentos longos

### 3. Organização melhor
- sair de uma pasta única por categoria para uma taxonomia em 2 níveis
- manter nomes consistentes por aliases/categorias canônicas
- criar áreas técnicas explícitas para revisão, falha e duplicados

### 4. Performance melhor
- separar pools por tipo de trabalho
- permitir múltiplos sub-processos/sub-agentes por estágio
- aplicar backpressure e limites de concorrência
- reduzir gargalo de inferência local

### 5. Produto melhor
- dry-run
- histórico reversível
- benchmark
- dataset de avaliação
- métricas por etapa
- configuração clara de modelos, thresholds e taxonomia

---

## Arquitetura proposta

### Pipeline novo
A app passa a operar em etapas explícitas:

1. `ingest`
2. `extract`
3. `understand`
4. `classify`
5. `act`
6. `review`

### Contrato interno principal
Criar um objeto estruturado, algo como `DocumentEnvelope`, contendo:
- `document_id`
- `source_path`
- `filename`
- `extension`
- `mime_type`
- `file_hash`
- `metadata`
- `extracted_text`
- `extraction_quality`
- `ocr_used`
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

Esse contrato será o backbone do sistema. Sem isso, a evolução fica acoplada e frágil.

---

## Estratégia de inteligência

### Modelo operacional
A escolha confirmada foi **local-only**.

Isso não significa “single-model”. A proposta é:

- `router_model`: triagem leve, tipo de documento, idioma, quick tags
- `understanding_model`: resumo e entendimento semântico
- `classification_model`: decisão de pasta
- `fallback_model`: opcional, ainda local, para casos difíceis

### Política de decisão
- regras fortes podem classificar diretamente
- LLM entra quando a regra não resolve
- se a extração estiver ruim, usar prompt específico de baixa evidência
- se a confiança for baixa, enviar para `_NeedsReview`
- criação de categorias novas deve passar por normalização contra taxonomia existente

### Prompts propostos
Separar prompts por responsabilidade:

#### Prompt 1: Understanding
Extrair:
- resumo objetivo
- idioma
- tipo documental
- temas centrais
- entidades
- palavras-chave

#### Prompt 2: Classification
Decidir:
- `category_l1`
- `category_l2`
- `confidence`
- `reason`
- `needs_review`

#### Prompt 3: Normalization
Normalizar:
- nome canônico de categoria
- alias conhecido
- nome seguro para filesystem

---

## Estratégia de formatos e extração

### Prioridade de cobertura
Fase inicial:
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

Fase seguinte:
- XLSX/XLS
- EML/MSG
- HTML
- JSON/XML
- ZIP como contêiner

### Melhorias obrigatórias
- MIME sniffing além de extensão
- OCR multipágina real
- metadados por extractor
- score de qualidade de extração
- fallback claro por tipo
- chunking inteligente por zonas prioritárias
- tratamento melhor de encoding

### Saída padrão de extractor
Cada extractor deve devolver:
- texto
- metadados
- qualidade
- páginas/slides/abas analisadas
- erros/fallbacks usados

---

## Estrutura de diretórios proposta

### Estrutura alvo
`organized_dir/<category_l1>/<category_l2>/arquivo.ext`

### Pastas técnicas
- `_NeedsReview`
- `_Unsupported`
- `_Duplicates`
- `_FailedExtraction`

### Regras
- `category_l1`: domínio amplo
- `category_l2`: subtema, tipo ou projeto
- usar aliases para convergir sinônimos
- evitar explosão de pastas com nomes quase idênticos
- nunca confiar cegamente no path original como verdade semântica

---

## Performance e paralelismo

### Problema atual
Hoje cada thread faz tudo de ponta a ponta:
- ler
- extrair
- chamar LLM
- mover
- registrar

Isso não distribui bem gargalos diferentes.

### Arquitetura proposta
Separar pools por estágio:
- pool I/O: leitura, hashing, descoberta
- pool CPU/subprocess: OCR, LibreOffice, parsing pesado
- pool LLM: inferência local limitada
- pool de persistência/move: operações finais

### Filas
- `ingest_queue`
- `extract_queue`
- `understand_queue`
- `classify_queue`
- `move_queue`

### Regras de operação
- OCR em subprocesso
- conversão `.doc/.ppt` em subprocesso isolado
- concorrência de LLM configurável e conservadora
- cache versionado por hash + prompt version + model version

---

## Confiabilidade e observabilidade

### Funcionalidades novas
- `dry-run`
- `undo-last-run`
- benchmark local
- dataset de avaliação
- logging estruturado por `run_id` e `document_id`
- métricas por etapa

### Métricas essenciais
- latência por estágio
- taxa de cache hit
- taxa de OCR
- taxa de review
- taxa de erro
- confiança média
- distribuição por categoria
- throughput por minuto

---

## Proposta de execução em etapas

### Etapa 1: Fundação arquitetural
Objetivo:
- modularizar o pipeline sem mudar radicalmente o comportamento externo

Entregas:
- criar módulos de pipeline
- introduzir `DocumentEnvelope`
- separar extração, entendimento, classificação e move
- organizar config para suportar evolução futura

Impacto:
- reduz acoplamento
- prepara terreno para todas as etapas seguintes

Estimativa de esforço:
- implementação: média
- risco: baixo
- consumo estimado de tokens no ChatGPT: **12k a 22k**

Observação:
- esta etapa cabe bem no plano gratuito se feita em 2 a 4 conversas curtas

---

### Etapa 2: Inteligência e prompts
Objetivo:
- melhorar a qualidade da decisão semântica

Entregas:
- quebrar o prompt monolítico em prompts específicos
- adicionar entendimento antes da classificação
- adicionar confidence/reason/needs_review
- adicionar regras determinísticas antes do LLM
- permitir múltiplos modelos locais configuráveis

Impacto:
- maior precisão
- menor inconsistência
- melhor comportamento em casos ambíguos

Estimativa de esforço:
- implementação: média/alta
- risco: médio
- consumo estimado de tokens: **18k a 35k**

Observação:
- esta é a etapa mais sensível a iteração de prompt
- vale dividir em subtarefas pequenas para não estourar limite do plano gratuito

---

### Etapa 3: Extração e formatos
Objetivo:
- melhorar a qualidade e a cobertura de ingestão de conteúdo

Entregas:
- extractor registry
- MIME sniffing
- score de qualidade
- OCR multipágina
- suporte a novos formatos prioritários
- chunking inteligente

Impacto:
- reduz arquivos “sem texto”
- melhora base para o classificador
- aumenta cobertura do produto

Estimativa de esforço:
- implementação: alta
- risco: médio
- consumo estimado de tokens: **16k a 30k**

Observação:
- pode ser dividida por família de formatos, o que ajuda bastante no plano gratuito

---

### Etapa 4: Taxonomia e estrutura de diretórios
Objetivo:
- tornar a organização final realmente útil e consistente

Entregas:
- estrutura em 2 níveis
- aliases/categorias canônicas
- criação de `_NeedsReview`, `_Unsupported`, `_Duplicates`, `_FailedExtraction`
- política de criação de novas categorias
- normalização de nomes de pastas

Impacto:
- aumenta valor percebido da app
- evita caos de diretórios
- melhora previsibilidade

Estimativa de esforço:
- implementação: média
- risco: médio
- consumo estimado de tokens: **10k a 20k**

Observação:
- etapa ótima para fazer no plano gratuito, pois é bem delimitada

---

### Etapa 5: Performance e sub-agentes/subprocessos
Objetivo:
- escalar melhor a execução local

Entregas:
- pools por estágio
- filas com backpressure
- concorrência separada para I/O, OCR e LLM
- isolamento de subprocessos pesados
- controle fino de workers

Impacto:
- melhor throughput
- menos contenção
- mais estabilidade em lotes grandes

Estimativa de esforço:
- implementação: alta
- risco: médio/alto
- consumo estimado de tokens: **18k a 32k**

Observação:
- esta etapa precisa de cuidado para não introduzir complexidade cedo demais
- idealmente vem depois das etapas 1 a 4

---

### Etapa 6: Benchmark, avaliação e confiabilidade
Objetivo:
- medir a app e permitir evolução sustentável

Entregas:
- dataset local de avaliação
- benchmark comparando prompts/modelos
- dry-run
- undo-last-run
- logs estruturados
- relatórios de qualidade e throughput

Impacto:
- reduz regressões
- permite otimizar com segurança
- melhora muito a manutenção

Estimativa de esforço:
- implementação: média
- risco: baixo
- consumo estimado de tokens: **12k a 24k**

Observação:
- mesmo não sendo “visível” para o usuário final, esta etapa é essencial para evoluir com segurança

---

## Estimativa total de tokens

### Cenário econômico para plano gratuito
Se dividirmos bem o trabalho:
- Etapa 1: 12k a 22k
- Etapa 2: 18k a 35k
- Etapa 3: 16k a 30k
- Etapa 4: 10k a 20k
- Etapa 5: 18k a 32k
- Etapa 6: 12k a 24k

Total acumulado aproximado:
- **mínimo**: 86k
- **máximo**: 163k

### Interpretação prática
Sim, **dá para evoluir no plano gratuito**, desde que a execução seja feita em conversas pequenas e focadas.

A estratégia recomendada é:
- uma conversa por subetapa
- nunca pedir “implemente tudo”
- trabalhar com escopo curto e validável
- pedir diffs pequenos
- manter contexto local no repositório, não no chat

---

## Estratégia recomendada para caber no plano gratuito

### Forma de trabalhar
Em vez de pedir uma grande refatoração, dividir em blocos como:
1. criar `DocumentEnvelope` e extrair pipeline em módulos
2. separar prompt de understanding
3. separar prompt de classification
4. adicionar confidence + `_NeedsReview`
5. criar taxonomia 2 níveis
6. refatorar extractors PDF/DOCX/PPTX
7. adicionar XLSX/EML
8. introduzir filas e pools por estágio
9. criar benchmark local
10. criar dry-run e undo

### Diretriz prática
Cada conversa deve:
- mexer em uma única fatia
- ter teste ou validação local
- atualizar documentação
- evitar reabrir todo o contexto do projeto

### Resultado
Assim o custo por conversa tende a ficar em algo como:
- **5k a 15k tokens** na maioria dos casos
- isso é muito mais seguro para o plano gratuito

---

## Ordem de implementação recomendada

### Ordem ideal
1. Fundação arquitetural
2. Prompting e intelligence split
3. Taxonomia 2 níveis
4. Extração e formatos
5. Review flow + confidence
6. Performance por estágio
7. Benchmark, dry-run e undo

### Ordem alternativa mais econômica
Se a prioridade for caber com folga no plano gratuito:
1. Fundação arquitetural mínima
2. Taxonomia 2 níveis
3. Confidence + `_NeedsReview`
4. Prompts separados
5. Melhorias graduais de extractors
6. Dry-run + benchmark
7. Performance avançada

Essa ordem entrega valor mais cedo com menor risco.

---

## Riscos principais
- aumentar a complexidade antes de estabilizar o pipeline
- criar taxonomia excessivamente livre e gerar sprawl
- saturar o modelo local com concorrência agressiva
- investir em prompts sem benchmark
- adicionar muitos formatos antes de padronizar a interface de extractor

---

## Critérios de sucesso
Vamos considerar a evolução bem-sucedida quando:
- houver melhora perceptível na precisão de classificação
- a estrutura final estiver consistente em 2 níveis
- arquivos ambíguos forem encaminhados corretamente para revisão
- a taxa de falha de extração cair
- o throughput em lotes maiores melhorar
- houver benchmark local para comparar mudanças
- a app continuar operável localmente sem depender de nuvem

---

## Conclusão
A app já tem uma base boa de MVP. O maior salto de valor agora não vem de “mais um modelo”, e sim de:

- modularizar o pipeline
- separar entendimento de classificação
- melhorar extração
- estruturar melhor a taxonomia
- introduzir revisão e métricas
- tratar performance por estágio

Se executarmos em fatias pequenas, esse roadmap é compatível com uso no plano gratuito do ChatGPT sem travar a evolução do projeto.
