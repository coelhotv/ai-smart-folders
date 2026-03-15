# AI Smart Folders Testing and Benchmark Guide

## Objetivo
Este manual explica como testar a app de forma controlada e como validar benchmarks reais no estado atual do projeto.

O foco e responder estas perguntas:
- a pipeline nova esta funcionando de ponta a ponta?
- os modelos atuais estao classificando bem?
- o OCR esta funcionando quando necessario?
- a taxonomia em 2 niveis faz sentido?
- o `dry-run` entrega informacao suficiente para revisao?
- a app esta segura para processar arquivos reais?

## Estado atual da app
Hoje o fluxo principal para testes e:

```bash
python main.py run
python main.py dry-run
python main.py benchmark --dataset evaluation/sample_dataset.jsonl
python main.py undo-last-run
python main.py reindex-taxonomy
```

Modelos atuais padrao:
- `router_model`: `granite4:1b-h`
- `understanding_model`: `qwen3.5:4b`
- `classification_model`: `qwen3.5:4b`
- `fallback_model`: `qwen3.5:9b-q4_K_M`
- `ocr_model`: `glm-ocr:q8_0`

## Antes de comecar

### 1. Confirmar que esta usando a venv certa
No projeto:

```bash
./.venv/bin/python --version
./.venv/bin/python - <<'PY'
from ai_smart_folders.config import load_config
cfg = load_config()
print(cfg.models)
PY
```

Voce deve ver os modelos acima.

### 2. Confirmar que o Ollama esta disponivel
Rode:

```bash
ollama list
```

E confirme que estes modelos existem:
- `granite4:1b-h`
- `qwen3.5:4b`
- `qwen3.5:9b-q4_K_M`
- `glm-ocr:q8_0`

### 3. Confirmar dependencias locais opcionais
Para cobertura melhor de testes:

```bash
which tesseract
which soffice
```

Se algum nao existir:
- sem `tesseract`, o fallback OCR antigo nao roda
- sem `soffice`, `.doc`, `.ppt` e `.xls` por conversao podem falhar

Isso nao impede todos os testes, mas muda o resultado esperado.

## Estrategia recomendada de testes
Nao teste tudo de uma vez.

A ordem recomendada e:
1. teste estrutural com `dry-run`
2. teste funcional com poucos arquivos reais
3. teste de OCR
4. teste de duplicatas
5. teste de benchmark com dataset
6. teste de `undo-last-run`

## Preparar um ambiente de teste seguro

### Estrutura recomendada
Crie pastas temporarias no proprio workspace:

```bash
mkdir -p tmp_inbox
mkdir -p tmp_output
mkdir -p tmp_eval_samples
```

Crie um arquivo de config temporario para testes:

```yaml
inbox_dir: ./tmp_inbox
organized_dir: ./tmp_output
data_dir: ./.ai-smart-folders-data
max_workers: 2
prompt_version: v3

models:
  router_model: granite4:1b-h
  understanding_model: qwen3.5:4b
  classification_model: qwen3.5:4b
  fallback_model: qwen3.5:9b-q4_K_M
  ocr_model: glm-ocr:q8_0

thresholds:
  review_confidence: 0.55

workers:
  io: 2
  llm: 1
  extract: 2
  act: 1

taxonomy:
  level1_default: General
  level2_default: Unsorted
```

Salve como:

```bash
tmp-test-config.yaml
```

## Que tipos de arquivos usar

### Lote minimo ideal para validacao manual
Monte um conjunto pequeno com 8 a 15 arquivos.

Idealmente inclua:
- 1 PDF textual
- 1 PDF escaneado
- 1 DOCX
- 1 PPTX
- 1 XLSX
- 1 EML
- 1 imagem JPG ou PNG com texto
- 1 TXT ou MD
- 1 arquivo duplicado de algum item acima
- 1 arquivo dificil ou ambiguo

### Que conteudo e melhor para testar
Prefira arquivos com classificacao esperada clara.

Exemplos bons:
- nota fiscal ou invoice
- curriculo ou resume
- apresentacao de projeto
- planilha financeira
- email comercial
- documento tecnico de codigo ou arquitetura
- foto de recibo

Evite inicialmente:
- arquivos enormes
- arquivos muito sensiveis
- arquivos com classificacao subjetiva demais
- formatos exóticos

## Onde copiar os arquivos
Copie os arquivos que quer validar para:

```bash
./tmp_inbox
```

Exemplo:

```bash
cp /caminho/do/arquivo1.pdf ./tmp_inbox/
cp /caminho/do/arquivo2.docx ./tmp_inbox/
```

Se quiser testar duplicata:

```bash
cp ./tmp_inbox/arquivo1.pdf ./tmp_inbox/arquivo1_copia.pdf
```

## Teste 1: validacao estrutural com dry-run

### Comando
```bash
./.venv/bin/python main.py dry-run --config tmp-test-config.yaml
```

### O que esperar
- a app nao deve mover nada
- deve retornar um JSON com metricas da execucao
- deve haver um array `documents`
- cada documento deve trazer:
  - `decision_source`
  - `category_l1`
  - `category_l2`
  - `confidence`
  - `needs_review`
  - `reason`
  - `destination_path`

### Como validar
Para cada arquivo:
- a categoria faz sentido?
- o destino sugerido faz sentido?
- `needs_review` aparece nos casos ambiguos?
- OCR foi usado quando esperava?
- `decision_source` bate com o comportamento esperado?

### Sinais bons
- invoices vao para algo como `Finance/Invoices`
- resumes vao para algo como `Career/Resumes`
- slides vao para `Knowledge/Presentations`
- duplicatas vao para `_Duplicates`
- arquivos sem texto vao para `_FailedExtraction` ou `_NeedsReview`

## Teste 2: execucao real

### Comando
```bash
./.venv/bin/python main.py run --config tmp-test-config.yaml
```

### O que esperar
- os arquivos saem de `tmp_inbox`
- os arquivos aparecem em `tmp_output`
- a estrutura deve refletir a taxonomia em 2 niveis
- casos tecnicos devem ir para:
  - `_NeedsReview`
  - `_Duplicates`
  - `_FailedExtraction`

### Como validar
Rode:

```bash
find tmp_output -maxdepth 3 -type f | sort
```

Verifique:
- os arquivos foram movidos
- os destinos batem com o `dry-run`
- nao houve sobrescrita indevida
- arquivos com mesmo nome receberam sufixo se necessario

## Teste 3: undo

### Comando
```bash
./.venv/bin/python main.py undo-last-run --config tmp-test-config.yaml
```

### O que esperar
- os arquivos voltam para a origem
- `tmp_output` perde os arquivos movidos na ultima run

### Como validar
```bash
find tmp_inbox -maxdepth 1 -type f | sort
find tmp_output -maxdepth 3 -type f | sort
```

## Teste 4: OCR

### Arquivos ideais
- 1 JPG com texto legivel
- 1 PNG com texto
- 1 PDF escaneado

### Objetivo
Verificar se:
- `glm-ocr:q8_0` esta sendo usado
- a extração nao volta vazia
- o destino final faz sentido

### Como verificar
Rode `dry-run` e observe em `documents`:
- `ocr_used: true`
- `reason`
- `destination_path`

Se o OCR multimodal falhar, a app pode:
- cair para `tesseract`
- ou mandar o documento para review/failed extraction

## Teste 5: duplicatas

### Como preparar
Use um arquivo repetido no inbox.

### O que esperar
- o arquivo duplicado deve ser detectado por hash
- ele deve ir para `_Duplicates`
- `decision_source` deve indicar duplicate
- `confidence` deve vir alta

## Teste 6: benchmark com dataset

### Objetivo
Comparar o que a app decidiu com o que voce esperava.

### Formato do dataset
Use JSONL, um caso por linha:

```json
{"source_path":"./tmp_eval_samples/invoice.pdf","expected_category_l1":"Finance","expected_category_l2":"Invoices","expected_needs_review":false}
{"source_path":"./tmp_eval_samples/resume.docx","expected_category_l1":"Career","expected_category_l2":"Resumes","expected_needs_review":false}
```

### Onde colocar os arquivos
Coloque os arquivos reais do benchmark em:

```bash
./tmp_eval_samples
```

Crie um dataset, por exemplo:

```bash
./evaluation/my_real_dataset.jsonl
```

### Comando
```bash
./.venv/bin/python main.py benchmark --config tmp-test-config.yaml --dataset evaluation/my_real_dataset.jsonl
```

### O que esperar
Um JSON com:
- `total_cases`
- `matched_level1`
- `matched_level2`
- `matched_review_flag`
- `full_matches`
- `failures`
- `cases`

### Como validar
Olhe:
- quantos acertaram `category_l1`
- quantos acertaram `category_l2`
- quantos marcaram review corretamente
- quais casos falharam e por que

### Meta inicial razoavel
Para um dataset pequeno e bem escolhido:
- `category_l1` com acerto alto
- `category_l2` bom, mas nao perfeito
- poucos falsos positivos de review

## Ordem recomendada de uso dos comandos

### Primeiro ciclo
```bash
./.venv/bin/python main.py dry-run --config tmp-test-config.yaml
./.venv/bin/python main.py run --config tmp-test-config.yaml
./.venv/bin/python main.py undo-last-run --config tmp-test-config.yaml
```

### Segundo ciclo
```bash
./.venv/bin/python main.py benchmark --config tmp-test-config.yaml --dataset evaluation/my_real_dataset.jsonl
```

## O que pode dar errado

### 1. Arquivo vai para `_NeedsReview`
Possiveis causas:
- confidence baixa
- texto ambiguo
- classificacao nao confiavel

O que fazer:
- revisar o `reason`
- olhar o tipo de arquivo
- verificar se o texto extraido foi suficiente

### 2. Arquivo vai para `_FailedExtraction`
Possiveis causas:
- extractor nao suportou bem o formato
- OCR falhou
- arquivo vazio ou corrompido

O que fazer:
- testar o mesmo arquivo sozinho
- conferir dependencias locais
- verificar se o arquivo realmente tem texto extraivel

### 3. OCR nao funciona
Possiveis causas:
- `glm-ocr:q8_0` nao acessivel no Ollama
- `pdf2image` ausente para PDFs escaneados
- imagem/PDF muito ruim
- fallback `tesseract` indisponivel

O que fazer:
- confirmar `ollama list`
- testar com imagem mais simples
- confirmar `tesseract`
- confirmar `pdf2image` se o caso for PDF escaneado

### 4. Benchmark retorna `missing`
Possiveis causas:
- caminho do arquivo no dataset esta errado
- arquivo nao existe
- caminho relativo foi montado errado

O que fazer:
- conferir `source_path`
- preferir caminhos relativos ao repo ou absolutos consistentes

### 5. Config errada ou modelos antigos ainda carregando
Possiveis causas:
- a CLI esta lendo outro arquivo de config
- `AI_SMART_CONFIG_PATH` esta setado no shell

O que fazer:
```bash
./.venv/bin/python - <<'PY'
from ai_smart_folders.config import default_config_path, load_config
print(default_config_path())
print(load_config().models)
PY
```

## Debug recomendado

### Ver qual config esta em uso
```bash
./.venv/bin/python - <<'PY'
from ai_smart_folders.config import default_config_path
print(default_config_path())
PY
```

### Ver modelos efetivos carregados
```bash
./.venv/bin/python - <<'PY'
from ai_smart_folders.config import load_config
print(load_config().models)
PY
```

### Ver logs
Os logs ficam em:

```bash
./.ai-smart-folders-data/agent.log
./.ai-smart-folders-data/api.log
```

### Ver banco SQLite
Arquivo:

```bash
./.ai-smart-folders-data/file_organization.db
```

Inspecao rapida:

```bash
sqlite3 ./.ai-smart-folders-data/file_organization.db "select run_id, started_at, completed_at, total_files, processed_files, failed_files from runs order by started_at desc limit 5;"
```

### Ver se os modelos estao instalados
```bash
ollama list
```

## Checklist de validacao final

Marque como pronto quando:
- `dry-run` listar corretamente todos os arquivos e destinos sugeridos
- `run` mover os arquivos para destinos plausiveis
- `undo-last-run` restaurar corretamente
- duplicatas irem para `_Duplicates`
- arquivos escaneados usarem OCR quando necessario
- benchmark rodar com dataset proprio sem `missing`
- a maioria dos casos esperados acertar `category_l1`

## Recomendaçao pratica
Comece com 8 a 10 arquivos muito claros.
So depois aumente a dificuldade.

Se voce tentar validar tudo com um lote grande e heterogeneo logo no inicio, fica mais dificil distinguir:
- erro de extractor
- erro de OCR
- erro de taxonomia
- erro de prompt/modelo

O melhor caminho e:
1. poucos arquivos
2. `dry-run`
3. `run`
4. `undo`
5. benchmark pequeno
6. so depois escalar
