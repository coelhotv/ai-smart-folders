# AI Smart Folders - Architecture & Benchmarking Learnings

Este documento consolida os aprendizados técnicos, bugs críticos solucionados e refatorações realizadas na infraestrutura do `ai-smart-folders` para otimizar o fluxo de OCR e LLM em hardware local restrito (16GB RAM) e habilitar um benchmarking estéril ("zero-cache").

## 1. O Problema do "Cache Fantasma" (Path Resolution)
Durante tentativas de resetar a aplicação para iniciar um novo benchmark sem vícios pregressos ("nuclear reset"), percebemos que arquivos continuavam extraindo de forma instantânea e puxando resultados antigos.
- **Root Cause:** A diretiva `data_dir` dentro do `AppConfig` (`models.py`) estava definida como `Path.home() / ".ai-smart-folders-data"`. Isso significa que, independentemente da pasta do repositório no Github (ou iCloud CloudDocs), a base de dados (`file_organization.db`) e os arquivos binários do cache (`.pkl`) eram gravados no **diretório raiz do usuário**.
- **Resolução:** Alterado para usar `Path.cwd() / ".ai-smart-folders-data"`. O ambiente agora é totalmente auto-contido e previsível, mantendo o `.db` e os `.pkl` na mesma pasta onde a aplicação/CLI é acionada. Caches antigos na raiz do usuário foram deletados manualmentes e os dados finalmente pararam de "ressuscitar". 

## 2. Separação de Caches: OCR vs Classificação
Antes, qualquer execução do pipeline (`organize` ou `benchmark`) misturava os passos. Quando mudávamos o LLM de Classificação (Ex: Granite para Qwen), a pipeline refazia a extração da imagem na GPU (Deepseek OCR).
- **Refatoração:** Introduzido o **`ExtractionCache`**. Agora a etapa pesada de visão computacional/extração roda isoladamente apenas 1 vez (armazenando dezenas de kbs em `extract_cache.pkl`).
- **Benefício:** Para rodar testes de acurácia de texto comparando Modelos Pequenos (SLMs), basta expurgar o `file_cache.pkl` e a base `SQLite` enquanto preservamos o `extract_cache.pkl`. As execuções levam minutos (e não horas), salvando desgaste da GPU local. 

## 3. Ollama Vision Exceptions (Extractors)
Imagens passaram a retornar instantaneamente (em vez de passar pelo Deepseek). Detectamos que o `extractors.py` possuía um bloco `except Exception:` genérico que retornava `None` em caso de erro na chamada do cliente do Ollama python com o parâmetro de array de imagens (`images: [image_path]`), causando um fallback silencioso na pipeline.
- Sem extração óptica legível, a pipeline alimentava blocos de texto vazios para os modelos de classificação, engasgando a acurácia global.
- O diagnóstico do trace oculto garantiu segurança ao depurar as incompatibilidades da API de visão.

## 4. O Viés do Início (The "Force Fit" Prompt)
No primeiro benchmark verdadeiramente purificado de cache, o **Granite 4:3B** leu um PDF acadêmico sobre inteligência artificial do MIT como: *Art & Design / Decorative Wall Hangings* (A mesma classificação de um pôster usado no 1º processo da fila).  
- **Root Cause:** Como a `existing_taxonomy` estava inicialmente vazia, a 1ª imagem determinou a única taxonomia disponível no universo da aplicação. A instrução do prompt forçava gentilmente (e matematicamente) os modelos pequenos a evitar a criação excessiva de pastas, causando um *overfitting* mortal em relação à única via disponível.
- **Resolução:** Refatoração do classification prompt em `llm.py` via F-Strings, fixando *Critical Rules*:
    1. A taxonomia legada só pode ser utilizada se a equivalência semântica for **altamente compatível (>80% relevância)**.
    2. Priorização severa em cima dos `Keywords` extraídos.
    3. Permissão estrita e explícita para *criar categorias novas* do zero caso não haja um fit exato com pastas herdadas.

## 5. Pydantic Structured Outputs (Ollama v0.2+)
Scripts antigos dependiam de manipulação de JSON sujo que falhava aleatoriamente em SLMs muito prolixos.
- **Resolução:** Integrada a funcionalidade `model_json_schema()` proveniente das estruturas Pydantic com o setter `format` nativo do cliente do Ollama (`_chat_json` dentro de `llm.py`). A saída das APIs de Small Models agora é contínua e formatada com rigor sem dependência de RegEx agressivo.

---

### Condução de Testes de Limpeza Recomendada (Tactic Reset):
Se o intuito futuro for validar uma nova classe sem descartar o processamento pesado já executado:
```bash
# Apague apenas a classificação e organização de arquivos:
rm ./.ai-smart-folders-data/file_cache.pkl ./.ai-smart-folders-data/file_organization.db

# Remonte os benchmarks comparativos em lote garantindo que eles absorvam os JSONLs do extraction_cache.pkl:
source .venv/bin/activate && python3 evaluation/run_benchmark.py --configs test-granite.yaml test-qwen.yaml test-gemma.yaml --dataset evaluation/ocr_benchmark_folders.jsonl
```

## 6. Integração do OpenDataLoader (Extração de PDFs em Lote)
A grande evolução na capacidade de leitura de PDFs complexos foi a adoção do backend **OpenDataLoader PDF (ODL)** via HTTP (`ai_smart_folders/odl.py`).
- **O Problema Original:** O Python puro usando `PyPDF2` deformava a estrutura de estudos acadêmicos e PDFs com colunas e tabelas multiplas. O "plano B" era jogar as páginas inteiras na GPU (`deepseek-ocr`), o que destruiria a performance em uma base com 5 mil arquivos.
- **A Solução (Hybrid Mode):** Nós recheamos a infraestrutura com a habilidade de consultar um microserviço JVM nativo que processa Lotes Inteiros (`extractors.py` recebe de mãos beijadas em Markdown). Os scripts de orquestração se encontram na pasta `scripts/` (ex: `./scripts/start-hybrid.sh`).
- **Ganhos Críticos:** 
    1. Respeito irrestrito ao `reading-order` semântico e recuperação perfeita de tabelas via algoritmo *XY-Cut++*.
    2. Evitou esgotamento da GPU em arquivos longos, reservando o OCR local focado em hardware apenas para imagens verdadeiras (`.jpg, .png`).
