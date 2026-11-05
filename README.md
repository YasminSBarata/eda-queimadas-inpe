# EDA orientada a hipóteses — Focos de calor do INPE no Databricks Free

Material da aula da eletiva *Plataformas de Dados e IA com Databricks* (CESUPA, 2026.2).

## Arquivos

| Arquivo | Papel | Quem roda |
| --- | --- | --- |
| `01_bronze_focos_inpe.py` | Ingestão dos CSVs mensais do INPE para `workspace.queimadas.bronze_focos_inpe` (texto puro, com proveniência) | Todos, antes da aula |
| `02_silver_focos_inpe.py` | Limpeza e enriquecimento → `silver_focos_inpe` + dicionário; reconciliação e 6 checagens | Todos, antes da aula |
| `03_eda_orientada_hipoteses_exemplo.py` | Exemplo guiado: 4 hipóteses pré-registradas (temporal, espacial, associativa, instrumento) com veredito calculado | Professor, ao vivo |
| `04_eda_alunos_template.py` | Template da equipe: pré-registro, sementes de hipótese, blocos por hipótese, síntese, checklist | Equipes |

O roteiro pedagógico (framework, exemplo comentado, enunciado, rubrica, armadilhas) está no documento *Aula — EDA Orientada a Hipóteses*.

## Como importar no Databricks

1. **Workspace → Import** → arraste os `.py` (formato *Source*). Cada um vira um notebook com células.
2. Rode `01` → `02` → `03`. Os widgets no topo definem catálogo (`workspace`), schema (`queimadas`), meses e UF.
3. Os CSVs vêm de `https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/mensal/Brasil/` (`focos_mensal_br_AAAAMM.csv`).
   Na Free Edition o download direto só funciona com internet liberada (verificação via LinkedIn). Caso contrário:
   baixe os arquivos no computador e suba em **Catalog → workspace → queimadas → raw → inpe_focos** (Upload to this volume).
   Ano de 2025 inteiro ≈ 520 MB; subconjunto leve `202501,202504,202507,202508,202509,202511` ≈ 320 MB.

## Fatos do dado que sustentam a aula

- Uma linha = uma detecção por satélite em uma passagem (não é um incêndio).
- 10+ satélites misturados; `AQUA_M-T` é o de referência do INPE para séries comparáveis.
- `lat`/`lon` com espaço antes do sinal; `-999` como sentinela; `frp` vazio nos METOP; horário em UTC.
- A Silver mantém todos os satélites e expõe `satelite_referencia`, `familia_satelite`, `amazonia_legal`, `faixa_dias_sem_chuva`, hora local (UTC−3, simplificação declarada).
