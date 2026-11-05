# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 · EDA orientada a hipóteses — exemplo guiado (professor)
# MAGIC
# MAGIC Este notebook é o **modelo de raciocínio**. Cada hipótese passa pelo mesmo ciclo:
# MAGIC
# MAGIC **Contexto → Hipótese (com direção) → Previsão mensurável → Recorte → Armadilha → Critério de veredito (ANTES de rodar) → Consulta → Gráfico → Veredito → Limitação / nova hipótese**
# MAGIC
# MAGIC Repare que a célula de código vem **depois** do pré-registro, nunca antes. A pergunta para a turma em cada etapa é:
# MAGIC *"O que eu precisaria ver para dizer que estou errado?"*
# MAGIC
# MAGIC ---
# MAGIC ## Passo 0 — Contexto e decisor
# MAGIC
# MAGIC **Decisor:** a Defesa Civil / brigadas de um estado da Amazônia Legal, planejando **quando** e **onde** posicionar equipes em 2026, com base no que aconteceu em 2025.
# MAGIC
# MAGIC **O que uma linha significa:** uma detecção de foco de calor por um satélite em uma passagem. Não é um incêndio; o mesmo fogo gera várias linhas.
# MAGIC
# MAGIC **Decisões já tomadas na Silver que afetam tudo:** `-999` virou `NULL`; hora local em UTC−3 para todo o país; `satelite_referencia` só para `AQUA_M-T`.

# COMMAND ----------

dbutils.widgets.text("catalogo", "workspace", "Catálogo")
dbutils.widgets.text("schema", "queimadas", "Schema")
dbutils.widgets.text("ano", "2025", "Ano analisado")
dbutils.widgets.text("uf_foco", "PA", "UF do decisor")

CATALOGO = dbutils.widgets.get("catalogo").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ANO = int(dbutils.widgets.get("ano"))
UF = dbutils.widgets.get("uf_foco").strip().upper()

from pyspark.sql import functions as F
import plotly.express as px

spark.table(f"{CATALOGO}.{SCHEMA}.silver_focos_inpe").filter(F.col("ano") == ANO).createOrReplaceTempView("focos")
print(f"Linhas na Silver ({ANO}):", spark.table("focos").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 1 — Entender o dado antes de hipotetizar (10 minutos, sem gráficos)
# MAGIC
# MAGIC Três consultas e só. O objetivo é conhecer **a granularidade e a cobertura**, não "explorar".

# COMMAND ----------

# 1a. Quais satélites existem e quanto cada um pesa? (se somarmos tudo, quem domina a contagem?)
display(spark.sql("""
    SELECT familia_satelite, satelite, COUNT(*) AS focos,
           ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_do_total,
           ROUND(AVG(frp_mw), 1) AS frp_medio_mw,
           SUM(CASE WHEN frp_mw IS NULL THEN 1 ELSE 0 END) AS frp_nulos
    FROM focos
    GROUP BY familia_satelite, satelite
    ORDER BY focos DESC
"""))

# COMMAND ----------

# 1b. Cobertura temporal: há meses faltando? O satélite de referência está em todos?
display(spark.sql("""
    SELECT mes, COUNT(*) AS todos_satelites,
           SUM(CASE WHEN satelite_referencia THEN 1 ELSE 0 END) AS aqua_mt,
           COUNT(DISTINCT data) AS dias_com_dado
    FROM focos GROUP BY mes ORDER BY mes
"""))

# COMMAND ----------

# 1c. O que está faltando nas medidas? (isso limita quais hipóteses podemos testar)
display(spark.sql("""
    SELECT
      ROUND(100 * AVG(CASE WHEN numero_dias_sem_chuva IS NULL THEN 1 ELSE 0 END), 1) AS pct_dias_sem_chuva_nulo,
      ROUND(100 * AVG(CASE WHEN risco_fogo IS NULL THEN 1 ELSE 0 END), 1) AS pct_risco_fogo_nulo,
      ROUND(100 * AVG(CASE WHEN frp_mw IS NULL THEN 1 ELSE 0 END), 1) AS pct_frp_nulo,
      ROUND(100 * AVG(CASE WHEN coord_valida THEN 0 ELSE 1 END), 2) AS pct_coord_invalida
    FROM focos
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC > **Pergunta para a turma após o Passo 1:** com o que vimos, que hipóteses *não* conseguimos testar com este dado sozinho?
# MAGIC > (Ex.: "município X tem mais fogo *por km²*" — não temos área. "O fogo causou desmatamento" — não temos desmatamento.)
# MAGIC
# MAGIC ---
# MAGIC ## Passo 2 — Pré-registro das hipóteses (preenchido ANTES de qualquer consulta)
# MAGIC
# MAGIC | # | Tipo | Hipótese | Previsão mensurável | Recorte | Critério de veredito | Armadilha e neutralização |
# MAGIC | --- | --- | --- | --- | --- | --- | --- |
# MAGIC | H1 | Temporal | Os focos concentram-se na estação seca | ago+set+out ≥ 50% dos focos do ano | Brasil, só `satelite_referencia` | Confirma ≥ 50%; refuta < 40%; inconclusiva entre | Somar todos os satélites muda o peso dos meses (satélites entram/saem do sistema). Neutralizar: só AQUA_M-T |
# MAGIC | H2 | Espacial | O fogo é concentrado: poucos municípios respondem pela maioria dos focos da UF | Top-10 municípios ≥ 50% dos focos da UF | UF do decisor, só referência | Confirma ≥ 50%; refuta < 35% | Sem denominador (área), "muito foco" pode ser "município enorme". Neutralizar: declarar a limitação e marcar como extensão o join com área IBGE |
# MAGIC | H3 | Associativa | Quanto mais dias sem chuva, mais intenso o fogo | Mediana de FRP cresce monotonicamente por faixa de dias sem chuva; faixa "31+" ≥ 1,3× a faixa "0-2" | Um único satélite por vez (AQUA_M-T e NOAA-20) | Confirma se monotônica e razão ≥ 1,3 nos dois satélites; refuta se não cresce; inconclusiva se só em um | Misturar sensores (GOES lê ~100 MW onde AQUA lê ~30); sentinela -999 puxando a média. Neutralizar: por satélite, mediana, Silver já anulou -999 |
# MAGIC | H4 | Instrumento | "A maioria dos focos ocorre à tarde" é fenômeno, não artefato de horário de passagem | A distribuição por hora local é parecida entre famílias de satélite | Brasil, por `familia_satelite` | Confirma se o pico da tarde aparece em **todas** as famílias; refuta/inconclusiva se o pico muda com o satélite | Satélites polares passam em horário fixo; só o GOES (geoestacionário) observa continuamente. Neutralizar: olhar o GOES separado |

# COMMAND ----------

# MAGIC %md
# MAGIC ## H1 · Temporal — "os focos concentram-se na estação seca"
# MAGIC **Gráfico escolhido:** barras por mês (uma série = uma medida ao longo do tempo).

# COMMAND ----------

# H1 — distribuição mensal, SÓ satélite de referência (o pré-registro exige isso)
h1 = spark.sql("""
    SELECT mes, COUNT(*) AS focos,
           ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM focos
    WHERE satelite_referencia
    GROUP BY mes ORDER BY mes
""")
display(h1)   # troque a visualização para "Bar" (eixo X = mes, Y = pct)

# COMMAND ----------

# H1 — veredito calculado contra o critério pré-registrado
pct_seca = h1.filter(F.col("mes").isin(8, 9, 10)).agg(F.sum("pct")).first()[0]
veredito = "CONFIRMADA" if pct_seca >= 50 else ("REFUTADA" if pct_seca < 40 else "INCONCLUSIVA")
print(f"H1 — ago+set+out = {pct_seca:.1f}% dos focos do ano (satélite de referência) → {veredito}")

# COMMAND ----------

# H1 — a ARMADILHA, mostrada de propósito: o mesmo cálculo somando todos os satélites
h1_todos = spark.sql("""
    SELECT mes, ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_todos
    FROM focos GROUP BY mes ORDER BY mes
""")
pct_seca_todos = h1_todos.filter(F.col("mes").isin(8, 9, 10)).agg(F.sum("pct_todos")).first()[0]
print(f"Mesma pergunta somando TODOS os satélites: {pct_seca_todos:.1f}%  (diferença de {pct_seca_todos - pct_seca:+.1f} pontos)")
display(h1.join(h1_todos, "mes").select("mes", "pct", "pct_todos").orderBy("mes"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H1.** O veredito vale para a *métrica declarada*. Se a diferença entre "só referência" e "todos" for grande,
# MAGIC o que mudou não foi o fogo — foi o sistema de observação. Isso é um achado sobre o **dado**, e vira a hipótese H4.
# MAGIC
# MAGIC **Limitação registrada:** um ano só. Sazonalidade exige comparar vários anos (extensão: baixar 2024).
# MAGIC
# MAGIC ---
# MAGIC ## H2 · Espacial — "poucos municípios concentram a maioria dos focos da UF"
# MAGIC **Gráfico escolhido:** barras ordenadas + curva acumulada (Pareto), e um mapa só para olhar *onde*.

# COMMAND ----------

# H2a — a UF do decisor está entre as mais afetadas do país? (contexto, não é a hipótese)
display(spark.sql("""
    SELECT uf, regiao, COUNT(*) AS focos,
           RANK() OVER (ORDER BY COUNT(*) DESC) AS posicao
    FROM focos WHERE satelite_referencia
    GROUP BY uf, regiao ORDER BY focos DESC LIMIT 10
"""))

# COMMAND ----------

# H2b — Pareto dos municípios da UF (só referência)
h2 = spark.sql(f"""
    WITH m AS (
        SELECT municipio, cod_ibge_municipio, COUNT(*) AS focos
        FROM focos WHERE satelite_referencia AND uf = '{UF}'
        GROUP BY municipio, cod_ibge_municipio
    ),
    r AS (
        SELECT *, ROW_NUMBER() OVER (ORDER BY focos DESC) AS posicao, SUM(focos) OVER () AS total_uf
        FROM m
    )
    SELECT posicao, municipio, cod_ibge_municipio, focos,
           ROUND(100 * focos / total_uf, 1) AS pct,
           ROUND(100 * SUM(focos) OVER (ORDER BY posicao) / total_uf, 1) AS pct_acumulado
    FROM r ORDER BY posicao
""")
display(h2.limit(20))

n_municipios = h2.count()
pct_top10 = h2.filter(F.col("posicao") == 10).select("pct_acumulado").first()[0]
veredito = "CONFIRMADA" if pct_top10 >= 50 else ("REFUTADA" if pct_top10 < 35 else "INCONCLUSIVA")
print(f"H2 — {UF}: top-10 de {n_municipios} municípios = {pct_top10:.1f}% dos focos → {veredito}")

# COMMAND ----------

# H2c — mapa: onde estão os focos de referência da UF (amostra para não pesar o navegador)
pts = (spark.table("focos")
       .filter(f"satelite_referencia AND uf = '{UF}' AND coord_valida")
       .select("lat", "lon", "mes", "municipio", "frp_mw"))
amostra = pts.sample(fraction=min(1.0, 20000 / max(pts.count(), 1)), seed=42).toPandas()

try:
    fig = px.scatter_mapbox(amostra, lat="lat", lon="lon", color="mes", hover_name="municipio",
                            zoom=4.5, height=600, opacity=0.5, color_continuous_scale="YlOrRd",
                            title=f"Focos (AQUA_M-T) em {UF}, {ANO} — amostra de {len(amostra):,} pontos")
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
except Exception:  # versão do plotly sem mapbox: dispersão simples lon x lat
    fig = px.scatter(amostra, x="lon", y="lat", color="mes", hover_name="municipio", opacity=0.4, height=600,
                     title=f"Focos (AQUA_M-T) em {UF}, {ANO} — amostra")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H2.** Os municípios no topo do Pareto costumam ser **gigantes em área** (no Pará: Altamira, São Félix do Xingu,
# MAGIC Novo Progresso). "Mais focos" aqui não é "mais fogo por km²". O pré-registro já previa isso — por isso a conclusão
# MAGIC honesta é: *concentração confirmada em termos absolutos; densidade inconclusiva sem o denominador.*
# MAGIC
# MAGIC **Extensão (Gold):** juntar a área territorial do IBGE pelo `cod_ibge_municipio` e refazer o ranking em focos/1000 km².
# MAGIC
# MAGIC ---
# MAGIC ## H3 · Associativa — "quanto mais dias sem chuva, mais intenso o fogo (FRP)"
# MAGIC **Gráfico escolhido:** barras por faixa de dias sem chuva, uma série por satélite (mediana, não média).

# COMMAND ----------

# H3 — mediana de FRP por faixa de seca, POR SATÉLITE (pré-registro: AQUA_M-T e NOAA-20)
ordem = ["0-2 dias", "3-7 dias", "8-15 dias", "16-30 dias", "31+ dias"]
h3 = spark.sql("""
    SELECT satelite, faixa_dias_sem_chuva, COUNT(*) AS n,
           ROUND(percentile_approx(frp_mw, 0.5), 1) AS frp_mediana_mw,
           ROUND(AVG(frp_mw), 1) AS frp_media_mw
    FROM focos
    WHERE frp_mw IS NOT NULL AND faixa_dias_sem_chuva <> 'sem info'
      AND satelite IN ('AQUA_M-T', 'NOAA-20')
    GROUP BY satelite, faixa_dias_sem_chuva
""")
h3p = h3.toPandas()
h3p["faixa_dias_sem_chuva"] = h3p["faixa_dias_sem_chuva"].astype("category").cat.set_categories(ordem, ordered=True)
h3p = h3p.sort_values(["satelite", "faixa_dias_sem_chuva"])
display(h3p)

fig = px.bar(h3p, x="faixa_dias_sem_chuva", y="frp_mediana_mw", color="satelite", barmode="group",
             title="H3 — FRP mediana por faixa de dias sem chuva (por satélite)", height=420)
fig.show()

# COMMAND ----------

# H3 — veredito automático contra o critério: monotônica + razão (31+)/(0-2) >= 1.3 nos dois satélites
resultados = {}
for sat, g in h3p.groupby("satelite"):
    serie = g.set_index("faixa_dias_sem_chuva")["frp_mediana_mw"].reindex(ordem)
    monotona = serie.is_monotonic_increasing
    razao = serie["31+ dias"] / serie["0-2 dias"] if serie["0-2 dias"] else float("nan")
    resultados[sat] = (monotona, razao)
    print(f"  {sat}: monotônica={monotona}, razão 31+/0-2 = {razao:.2f}")

ok = [m and r >= 1.3 for m, r in resultados.values()]
veredito = "CONFIRMADA" if all(ok) else ("REFUTADA" if not any(m for m, _ in resultados.values()) else "INCONCLUSIVA")
print("H3 →", veredito)

# COMMAND ----------

# H3 — a ARMADILHA: o mesmo gráfico misturando todos os satélites
h3_mix = spark.sql("""
    SELECT faixa_dias_sem_chuva, ROUND(percentile_approx(frp_mw, 0.5), 1) AS frp_mediana_todos
    FROM focos WHERE frp_mw IS NOT NULL AND faixa_dias_sem_chuva <> 'sem info'
    GROUP BY faixa_dias_sem_chuva
""").toPandas()
h3_mix["faixa_dias_sem_chuva"] = h3_mix["faixa_dias_sem_chuva"].astype("category").cat.set_categories(ordem, ordered=True)
display(h3_mix.sort_values("faixa_dias_sem_chuva"))
print("Pergunta: a composição de satélites muda entre as faixas? Se sim, o 'efeito da seca' pode ser efeito de sensor.")
display(spark.sql("""
    SELECT faixa_dias_sem_chuva, familia_satelite, COUNT(*) AS n
    FROM focos WHERE frp_mw IS NOT NULL AND faixa_dias_sem_chuva <> 'sem info'
    GROUP BY faixa_dias_sem_chuva, familia_satelite ORDER BY faixa_dias_sem_chuva, n DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H3.** Mesmo confirmada, é **associação**, não causa: dias sem chuva, bioma e estação andam juntos.
# MAGIC Nova hipótese derivada (para os alunos): *a relação se mantém dentro de um único bioma?*
# MAGIC
# MAGIC ---
# MAGIC ## H4 · Instrumento — "o pico da tarde é do fogo ou do satélite?"
# MAGIC **Gráfico escolhido:** linha por hora local, uma série por família de satélite.

# COMMAND ----------

# H4a — distribuição por hora local, todos os satélites juntos (o gráfico "ingênuo")
display(spark.sql("SELECT hora_local, COUNT(*) AS focos FROM focos GROUP BY hora_local ORDER BY hora_local"))

# COMMAND ----------

# H4b — a mesma distribuição, separada por família de satélite (linha; X = hora_local, Y = pct, série = familia)
h4 = spark.sql("""
    SELECT familia_satelite, hora_local, COUNT(*) AS focos,
           ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY familia_satelite), 1) AS pct_na_familia
    FROM focos GROUP BY familia_satelite, hora_local
""").toPandas().sort_values(["familia_satelite", "hora_local"])
fig = px.line(h4, x="hora_local", y="pct_na_familia", color="familia_satelite", markers=True, height=450,
              title="H4 — % dos focos por hora local, por família de satélite (UTC−3 para todo o Brasil)")
fig.show()

# COMMAND ----------

# H4 — veredito: em quantas famílias o pico fica entre 12h e 17h?
pico = h4.loc[h4.groupby("familia_satelite")["pct_na_familia"].idxmax(), ["familia_satelite", "hora_local", "pct_na_familia"]]
display(pico)
n_fam = len(pico)
n_tarde = int(pico["hora_local"].between(12, 17).sum())
horas_cobertas = h4[h4["focos"] > 0].groupby("familia_satelite")["hora_local"].nunique()
print("Horas do dia com detecções, por família:\n", horas_cobertas.to_string())
veredito = "CONFIRMADA (pico da tarde aparece em todas as famílias)" if n_tarde == n_fam else \
           f"INCONCLUSIVA/REFUTADA: pico da tarde em {n_tarde} de {n_fam} famílias; a hora depende do satélite"
print("H4 →", veredito)

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H4.** Satélites polares (AQUA, TERRA, NOAA, NPP, METOP) passam em horários fixos — o histograma deles é o
# MAGIC **horário da passagem**, não o ciclo do fogo. O GOES é geoestacionário e observa o dia inteiro: é a única família em que
# MAGIC o pico da tarde pode ser lido como fenômeno. Lição: *um padrão forte no gráfico pode ser o instrumento falando.*
# MAGIC
# MAGIC **Limitação registrada:** fuso único UTC−3; para AC/AM/RR/RO/MT/MS a hora local está deslocada.
# MAGIC
# MAGIC ---
# MAGIC ## Passo final — Síntese (o que o decisor leva)
# MAGIC
# MAGIC Preencha com os vereditos impressos acima.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Achado 1** | H1: __% dos focos do ano (satélite de referência) caem em ago–out → janela de posicionamento das brigadas |
# MAGIC | **Achado 2** | H2: top-10 municípios de __ concentram __% dos focos → lista nominal para priorizar (em termos absolutos) |
# MAGIC | **Achado 3** | H3: FRP mediana cresce com dias sem chuva (razão __ no AQUA, __ no NOAA-20) → dias sem chuva é um gatilho operacional |
# MAGIC | **Limitação 1** | Um foco ≠ um incêndio; contagens misturam sensores. Séries só com AQUA_M-T |
# MAGIC | **Limitação 2** | Sem denominador de área: concentração absoluta ≠ densidade |
# MAGIC | **Limitação 3** | Hora local aproximada (UTC−3); horários de polares refletem passagem, não fogo |
# MAGIC | **Próximo passo (Gold)** | `gold_focos_mes_uf` (só referência), `gold_pareto_municipio_uf`, join com área IBGE; baixar 2024 para confirmar sazonalidade |
# MAGIC
# MAGIC **Surpresas que viraram hipóteses novas:** a razão todos/referência muda ao longo do ano (H1 → H4); FRP por faixa de seca pode ser efeito de composição de sensores (H3 → "repetir dentro de um bioma").

# COMMAND ----------

# (Opcional) Exemplo de tabela Gold nascida de uma hipótese confirmada — só referência, pronta para dashboard
spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOGO}.{SCHEMA}.gold_focos_mes_uf AS
SELECT ano, mes, regiao, uf, amazonia_legal, COUNT(*) AS focos_referencia
FROM {CATALOGO}.{SCHEMA}.silver_focos_inpe
WHERE satelite_referencia
GROUP BY ano, mes, regiao, uf, amazonia_legal
""")
display(spark.table(f"{CATALOGO}.{SCHEMA}.gold_focos_mes_uf").orderBy(F.desc("focos_referencia")).limit(10))