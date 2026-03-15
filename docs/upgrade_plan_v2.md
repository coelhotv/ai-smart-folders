# AI Smart Folders: Plano de Evolução v2 (Stack Moderno)

## Objetivo
Evoluir o AI Smart Folders para uma plataforma local inteligente, eficiente e auditável, utilizando um stack Python moderno para orquestrar um pipeline de processamento de arquivos robusto e concorrente.

Este plano incorpora as recomendações da análise de stack, focando em **Ray** para orquestração, **Pydantic** para contratos de dados e **Typer** para a CLI.

---

## Visão alvo (Inalterada)
A visão de uma aplicação com melhor inteligência, eficiência, organização, performance e usabilidade continua a mesma. A mudança está em **como** chegaremos lá.

---

## Arquitetura Proposta (Stack Moderno)

### Pipeline Baseado em Atores com Ray
A aplicação será refatorada para um pipeline explícito onde cada estágio é executado por um ou mais **Atores Ray**. Atores são processos Python independentes, o que resolve o gargalo do GIL e permite o gerenciamento de pools de workers específicos para cada tipo de carga de trabalho.

1.  **`Ingestor Actor`**: Monitora o inbox e cria um `DocumentEnvelope` para cada novo arquivo.
2.  **`Extractor Pool`**: Um pool de atores Ray que executa tarefas de CPU (OCR, extração de texto, conversão de `.doc`). Cada ator processa um arquivo.
3.  **`Understander Actor`**: Um ou mais atores que recebem o texto extraído e utilizam um modelo de LLM (ex: `router_model`) para extrair metadados, resumo e palavras-chave.
4.  **`Classifier Actor`**: Um ator que utiliza os dados do `Understander` para tomar a decisão final de categorização (`category_l1`, `category_l2`).
5.  **`Action Pool`**: Um pool de atores de I/O que movem os arquivos, atualizam o banco de dados SQLite e gerenciam as pastas técnicas (`_NeedsReview`, etc.).

### Contrato Interno com Pydantic
O `DocumentEnvelope` será implementado como um modelo **Pydantic**.

**Benefícios:**
-   **Validação Automática:** Garante que os dados que fluem entre os atores do pipeline sejam sempre válidos e completos.
-   **Auto-documentação:** O código se torna a fonte da verdade para a estrutura dos dados.
-   **Serialização Confiável:** Pydantic lida com a conversão de/para JSON de forma robusta, essencial para a comunicação entre processos.

```python
# Exemplo de definição em Pydantic
from pydantic import BaseModel, FilePath
from typing import List, Optional

class DocumentEnvelope(BaseModel):
    document_id: str
    source_path: FilePath
    file_hash: str
    status: str = "ingested"
    
    # ... todos os outros campos do plano original
    
    extracted_text: Optional[str]
    category_l1: Optional[str]
    category_l2: Optional[str]
    confidence: Optional[float]
    needs_review: bool = False
```

### CLI Robusta com Typer
A interface de linha de comando será reescrita com **Typer** para facilitar a criação de comandos complexos e bem documentados.

```bash
# Exemplo de como a CLI poderia se parecer
python main.py organize --path /meu/inbox --dry-run
python main.py benchmark --model "novo_modelo"
python main.py undo --run-id "abc-123"
```

---

## Performance e Paralelismo com Ray

A seção anterior sobre "pools por estágio" e "filas" é agora concretizada pelo Ray.

### Arquitetura de Concorrência
-   **Problema Resolvido:** O modelo anterior (um thread por arquivo do início ao fim) cria contenção. Se 5 arquivos grandes estiverem fazendo OCR, as chamadas rápidas ao LLM para outros 10 arquivos pequenos ficam bloqueadas.
-   **Solução com Ray:**
    -   **Pool de Atores CPU-Bound:** Um pool de atores Ray (`num_cpus=...`) será dedicado a tarefas pesadas como OCR e conversão de formatos. Como são processos separados, eles utilizam todos os cores da CPU.
    -   **Pool de Atores I/O-Bound:** Atores para mover arquivos ou fazer chamadas de rede podem rodar com `num_cpus=0`, indicando que não precisam de um core dedicado.
    -   **Ator GPU-Bound (Opcional):** Um ator pode ser fixado a uma GPU (`num_gpus=1`) para gerenciar exclusivamente as chamadas ao LLM, evitando que múltiplas inferências sobrecarreguem a VRAM.
-   **Filas:** A comunicação entre os estágios será feita por **`ray.Queue`**, que são filas distribuídas e seguras para comunicação entre os atores.

---

## Proposta de Execução em Etapas (Ajustada)

A ordem das etapas permanece logicamente a mesma, mas o conteúdo da **Etapa 1** é radicalmente diferente e mais concreto.

### **Etapa 1: Fundação Arquitetural com Ray e Pydantic (MVP v2)**
**Objetivo:** Reconstruir o esqueleto da aplicação sobre o novo stack.

**Entregas:**
1.  **Setup do Projeto:**
    -   Criar a nova estrutura de módulos (`pipeline/`, `actors/`, `models/`).
    -   Adicionar `ray`, `pydantic`, e `typer` às dependências.
2.  **CLI com Typer:**
    -   Criar o ponto de entrada `main.py` com um comando básico `organize`.
3.  **Definir `DocumentEnvelope`:**
    -   Criar o modelo Pydantic em `models/envelope.py`.
4.  **Implementar o Pipeline com Atores Ray:**
    -   Criar um ator Ray simples para cada estágio do pipeline (`Ingestor`, `Extractor`, `Classifier`, `Action`).
    -   No início, a lógica dentro de cada ator pode ser a mesma da `smart-folders_v2.py`. O objetivo é ter o fluxo de dados passando pelos atores e filas.
5.  **Inicialização do Ray:**
    -   Adicionar `ray.init()` ao início do processo.

**Impacto:**
-   **Enorme.** Reduz o acoplamento drasticamente e cria uma base escalável. A aplicação funcionará de forma semelhante por fora, mas por dentro estará pronta para a evolução.

**Estimativa de Esforço:**
-   Implementação: **Alta**. Esta é a etapa mais complexa e que exige a maior refatoração.
-   Risco: **Médio**. Requer um bom entendimento de Ray, mas o resultado é uma base muito mais estável.

---

### Etapas 2 a 6 (Conteúdo Inalterado, mas mais fácil de implementar)

As etapas seguintes do plano original (`Inteligência e Prompts`, `Extração e Formatos`, etc.) permanecem as mesmas em seu objetivo, mas sua implementação se torna muito mais fácil e limpa sobre a nova arquitetura.

-   **Adicionar um modelo?** Crie um novo Ator Ray para ele.
-   **Suportar um novo formato?** Crie uma nova função e a chame dentro do pool de `Extractor`.
-   **Mudar a lógica de classificação?** Modifique apenas o `ClassifierActor` sem tocar no resto do pipeline.
-   **Implementar `dry-run`?** Adicione um `if dry_run:` no `ActionActor` para logar em vez de mover o arquivo.

---

## Conclusão Atualizada

O plano de evolução continua válido, mas a decisão de adotar um stack Python moderno com **Ray, Pydantic e Typer** torna a execução desse plano **mais concreta, robusta e menos arriscada a longo prazo**. A complexidade é movida para a **Etapa 1 (Fundação Arquitetural)**, mas em troca, todas as etapas subsequentes se tornam mais simples, seguras e fáceis de paralelizar, resultando em uma aplicação final muito superior.
