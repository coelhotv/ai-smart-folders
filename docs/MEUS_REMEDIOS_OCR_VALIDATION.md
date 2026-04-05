# Validação de OCR para Receituários e Recibos (App Meus Remédios)

## Objetivo
Comparar diretamente modelos visuais locais (`deepseek-ocr:3b` vs `glm-ocr:q8_0`) no cenário de extração pontual de chaves e valores a partir de fotos de notas fiscais e recibos de farmácias. 

O foco deste caso de uso não é gerar contexto textual puro, mas sim extrair entidades específicas (`medicamento` e `quantidade`) com alta confiabilidade para posterior uso na UI e banco de dados do paciente.

## Metodologia Sugerida

### 1. Preparação do Dataset de Ground Truth
Selecionar ~10 fotos reais de cupons fiscais.
Criar um arquivo `evaluation/receipts_ground_truth.json` mapeando a expectativa:

```json
{
  "receipt_01.jpg": {"remedio": "Dipirona 500mg", "quantidade": 2},
  "receipt_02.jpg": {"remedio": "Roacutan 20mg", "quantidade": 1}
}
```

### 2. Script de Benchmark Customizado
No repositório do "Meus Remédios" (ou em um script isolado), escrever um `benchmark_receipts.py` que realiza o seguinte fluxo para cada imagem no dataset:
- Faz a inferência direta no Ollama chamando primeiro `glm-ocr:q8_0` e depois `deepseek-ocr:3b`.
- Usa um prompt restrito para extração de dados Pydantic/JSON.
```text
Role: Você é um assistente de extração de dados fiscais. 
Tarefa: Encontre medicamentos e quantidades nesta nota fiscal da farmácia.
Output esperado em JSON: [{"remedio": "Nome...", "quantidade": X}]
```

### 3. Métricas Principais a Avaliar
- **Recall Exato do Nome**: O modelo consegue acertar os miligramas (mg/ml) presentes na nota?
- **Precisão na Quantidade**: O modelo confundiu o preço pela quantidade do medicamento?
- **Latência**: Tempo médio por foto (Sendo para ambiente mobile, latência > 10s pode frustrar a UX).

### 4. Trade-off e Escolha do Modelo
Se o modelo de 6.7GB (`deepseek-ocr`) se mostrar apenas marginalmente melhor que o de 1.6GB (`glm-ocr`), o de 1.6GB deve vencer devido à maior rapidez (UX Mobile) e menor custo computacional no dispositivo hospedeiro.
