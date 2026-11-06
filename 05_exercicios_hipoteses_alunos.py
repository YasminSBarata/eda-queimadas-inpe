# Databricks notebook source
# MAGIC %pip install plotly -q

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 · Lista de exercícios — cinco hipóteses para testar na Silver
# MAGIC
# MAGIC **Equipe:** _______________  **Integrantes:** _______________
# MAGIC
# MAGIC Vocês já têm a tabela `workspace.queimadas.silver_focos_inpe` (focos de calor do INPE, 2025). As cinco hipóteses abaixo
# MAGIC **já vêm formuladas**, com previsão mensurável e critério de veredito. O que não vem: qual consulta fazer, quais colunas usar,
# MAGIC que filtro aplicar, qual gráfico desenhar e o que pode enganar. Isso é o exercício.
# MAGIC
# MAGIC Regras:
# MAGIC 1. Antes de rodar qualquer consulta de uma hipótese, preencha a célula **"Antes de rodar"** dela: o que você precisa saber do dado e qual armadilha espera encontrar.
# MAGIC 2. Toda célula de código começa com um comentário dizendo qual hipótese ela testa. Célula sem hipótese é turismo.
# MAGIC 3. O veredito é calculado em código, contra o critério escrito — não "no olho" pelo gráfico.
# MAGIC 4. O critério não muda depois de ver o resultado. Se você acha que ele estava errado, diga isso na discussão e proponha o próximo.
# MAGIC 5. Cada hipótese termina com três linhas de discussão: o que o número diz, o que ele não diz, e que hipótese nova surgiu.
# MAGIC
# MAGIC Dica única, válida para as cinco: o dicionário da Silver está em `workspace.queimadas.silver_focos_inpe_dicionario`. Leia antes.

# COMMAND ----------

dbutils.widgets.text("catalogo", "workspace", "Catálogo")
dbutils.widgets.text("schema", "queimadas", "Schema")
dbutils.widgets.text("ano", "2025", "Ano analisado")
CATALOGO = dbutils.widgets.get("catalogo").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ANO = int(dbutils.widgets.get("ano"))

from pyspark.sql import functions as F
import plotly.express as px

spark.table(f"{CATALOGO}.{SCHEMA}.silver_focos_inpe").filter(F.col("ano") == ANO).createOrReplaceTempView("focos")
display(spark.table(f"{CATALOGO}.{SCHEMA}.silver_focos_inpe_dicionario"))

# COMMAND ----------

# Registro dos vereditos — preencha ao final de cada hipótese
VEREDITOS = {}

def registrar(codigo, metrica_observada, criterio, veredito, armadilha_verificada, limitacao=""):
    VEREDITOS[codigo] = dict(metrica=metrica_observada, criterio=criterio, veredito=veredito,
                             armadilha=armadilha_verificada, limitacao=limitacao)
    print(f"{codigo}: {metrica_observada} | critério: {criterio} → {veredito}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## H1 · Comparativa — "O Cerrado queima antes da Amazônia"
# MAGIC
# MAGIC **Hipótese:** o calendário do fogo não é o mesmo nos dois biomas: o Cerrado atinge o pico mensal de focos **antes** da Amazônia.
# MAGIC
# MAGIC **Previsão mensurável:** o mês de pico do Cerrado é pelo menos um mês anterior ao mês de pico da Amazônia.
# MAGIC
# MAGIC **Critério de veredito:** confirmada se o pico do Cerrado vier pelo menos 1 mês antes; refutada se vier no mesmo mês ou depois; inconclusiva se os dois biomas tiverem dois meses empatados no topo (diferença < 5% entre eles).
# MAGIC
# MAGIC **Antes de rodar** (preencha):
# MAGIC - O que é "um foco" para esta contagem e qual filtro você vai usar para comparar meses? ______
# MAGIC - Comparar contagens absolutas entre dois biomas de tamanhos diferentes responde à pergunta? O que você vai comparar? ______
# MAGIC - Que armadilha espera encontrar? ______

# COMMAND ----------

# H1 — consulta


# COMMAND ----------

# H1 — gráfico (qual forma responde a "quando é o pico"?)


# COMMAND ----------

# H1 — veredito calculado contra o critério
# registrar("H1", "pico Cerrado = mês __ | pico Amazônia = mês __", "Cerrado ≥ 1 mês antes", "CONFIRMADA | REFUTADA | INCONCLUSIVA", armadilha_verificada="...")

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H1 (3 linhas):**
# MAGIC
# MAGIC ---
# MAGIC ## H2 · Temporal — "Fogo é atividade humana: o fim de semana queima menos"
# MAGIC
# MAGIC **Hipótese:** boa parte dos focos vem de manejo e queima intencional, que seguem a semana de trabalho; sábados e domingos têm menos focos por dia que dias úteis.
# MAGIC
# MAGIC **Previsão mensurável:** a média de focos por dia nos fins de semana é pelo menos 15% menor que a média por dia nos dias úteis.
# MAGIC
# MAGIC **Critério de veredito:** confirmada se a diferença for ≥ 15%; refutada se for < 5% (ou se o fim de semana tiver mais); inconclusiva entre 5% e 15%.
# MAGIC
# MAGIC **Antes de rodar** (preencha):
# MAGIC - "Dia" aqui é definido por qual coluna, e em que fuso? Isso pode mudar o dia de um foco? ______
# MAGIC - Um satélite que passa todo dia no mesmo horário vê o fim de semana do mesmo jeito que o dia útil? Qual satélite usar? ______
# MAGIC - Que armadilha espera encontrar? ______

# COMMAND ----------

# H2 — consulta

# COMMAND ----------

# H2 — gráfico

# COMMAND ----------

# H2 — veredito
# registrar("H2", "média dia útil = __ | média fim de semana = __ | diferença = __%", "≥ 15% menor", "...", armadilha_verificada="...")

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H2 (3 linhas):**
# MAGIC
# MAGIC ---
# MAGIC ## H3 · Espacial — "Na estação seca o fogo se espalha, não só se intensifica"
# MAGIC
# MAGIC **Hipótese:** o aumento de focos na seca não vem só dos mesmos lugares queimando mais; vem de mais lugares queimando.
# MAGIC
# MAGIC **Previsão mensurável:** o número de municípios com pelo menos um foco no mês de pico é pelo menos o dobro do número de municípios com foco em maio.
# MAGIC
# MAGIC **Critério de veredito:** confirmada se a razão (municípios no pico ÷ municípios em maio) for ≥ 2; refutada se for < 1,3; inconclusiva entre.
# MAGIC
# MAGIC **Antes de rodar** (preencha):
# MAGIC - "Município com foco" muda se você contar todos os satélites ou só um? Em qual direção? ______
# MAGIC - Além de contar municípios, que segunda métrica diria se o fogo também se *concentrou*? ______
# MAGIC - Que armadilha espera encontrar? ______

# COMMAND ----------

# H3 — consulta

# COMMAND ----------

# H3 — gráfico

# COMMAND ----------

# H3 — veredito
# registrar("H3", "municípios em maio = __ | no pico = __ | razão = __", "razão ≥ 2", "...", armadilha_verificada="...")

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H3 (3 linhas):**
# MAGIC
# MAGIC ---
# MAGIC ## H4 · Associativa — "O risco previsto antecipa a intensidade"
# MAGIC
# MAGIC **Hipótese:** o índice de risco de fogo que o INPE calcula para o local tem relação com o quanto o fogo de fato queima: focos em locais de risco alto são mais intensos.
# MAGIC
# MAGIC **Previsão mensurável:** a intensidade mediana dos focos com risco ≥ 0,7 é pelo menos 30% maior que a dos focos com risco < 0,3.
# MAGIC
# MAGIC **Critério de veredito:** confirmada se a razão entre as medianas for ≥ 1,3 em pelo menos dois satélites analisados separadamente; refutada se for ≤ 1,0 em todos; inconclusiva nos demais casos.
# MAGIC
# MAGIC **Antes de rodar** (preencha):
# MAGIC - Que coluna mede "intensidade"? Ela existe para todos os satélites? Pode ser comparada entre eles? ______
# MAGIC - O que a Silver fez com os valores de risco que vieram como -999? Isso afeta quem entra na análise? ______
# MAGIC - O dado tem risco só onde houve foco. O que isso impede você de concluir, mesmo se a hipótese for confirmada? ______

# COMMAND ----------

# H4 — consulta

# COMMAND ----------

# H4 — gráfico

# COMMAND ----------

# H4 — veredito
# registrar("H4", "razão mediana alto/baixo: sat A = __, sat B = __", "≥ 1,3 em dois satélites", "...", armadilha_verificada="...")

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H4 (3 linhas):**
# MAGIC
# MAGIC ---
# MAGIC ## H5 · De instrumento — "A contagem total e a de referência contam a mesma história"
# MAGIC
# MAGIC **Hipótese:** somar todos os satélites só muda a escala da contagem, não o formato: a fração dos focos que vem do satélite de referência é estável ao longo do ano.
# MAGIC
# MAGIC **Previsão mensurável:** a participação do satélite de referência no total mensal varia menos de 10 pontos percentuais entre o mês em que é maior e o mês em que é menor.
# MAGIC
# MAGIC **Critério de veredito:** confirmada se a variação for < 10 pontos; refutada se for ≥ 15 pontos; inconclusiva entre.
# MAGIC
# MAGIC **Antes de rodar** (preencha):
# MAGIC - Se a hipótese for refutada, o que isso diz sobre qualquer gráfico de "focos por mês" que some todos os satélites? ______
# MAGIC - Há satélites que entram ou saem da base ao longo do ano? Como você descobriria? ______
# MAGIC - Que armadilha espera encontrar? ______

# COMMAND ----------

# H5 — consulta

# COMMAND ----------

# H5 — gráfico

# COMMAND ----------

# H5 — veredito
# registrar("H5", "participação mín = __% (mês __) | máx = __% (mês __) | variação = __ pontos", "< 10 pontos", "...", armadilha_verificada="...")

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H5 (3 linhas):**
# MAGIC
# MAGIC ---
# MAGIC ## Síntese da equipe

# COMMAND ----------

import pandas as pd
display(pd.DataFrame.from_dict(VEREDITOS, orient="index").reset_index().rename(columns={"index": "hipotese"}))

# COMMAND ----------

# MAGIC %md
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Hipótese que mais surpreendeu e por quê** | |
# MAGIC | **Armadilha que quase pegou a equipe** | |
# MAGIC | **Critério que a equipe mudaria, e para quê** | |
# MAGIC | **Hipótese nova que surgiu de uma refutação** | |
# MAGIC | **O que não dá para responder com esta base** | |
# MAGIC
# MAGIC ### Entrega
# MAGIC Este notebook executado do início ao fim (Run all), com as cinco células "Antes de rodar" preenchidas **antes** das consultas, os cinco vereditos registrados e a síntese.
# MAGIC Hipótese refutada com critério respeitado vale o mesmo que confirmada. Hipótese com critério alterado depois do resultado, sem declarar, vale zero.