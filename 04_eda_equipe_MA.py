# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %pip install plotly -q

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 · EDA orientada a hipóteses — Maranhão: Cerrado × Amazônia
# MAGIC
# MAGIC **Equipe:** _______________________  **Integrantes:** _______________________
# MAGIC
# MAGIC Regras do jogo (do template):
# MAGIC 1. Passos 0, 1 e 2 preenchidos **antes** de rodar as consultas das hipóteses.
# MAGIC 2. 3 a 5 hipóteses de tipos diferentes. Aqui: **comparativa, instrumento, espacial, associativa, temporal**.
# MAGIC 3. Cada célula de código começa com `# Hx — ...`.
# MAGIC 4. Veredito calculado contra o critério pré-registrado.
# MAGIC 5. Entrega: notebook executado (com saídas) + síntese preenchida.
# MAGIC
# MAGIC Dados: `workspace.queimadas.silver_focos_inpe` (dicionário em `silver_focos_inpe_dicionario`).

# COMMAND ----------

dbutils.widgets.text("catalogo", "workspace", "Catálogo")
dbutils.widgets.text("schema", "queimadas", "Schema")
dbutils.widgets.text("ano", "2025", "Ano analisado")
dbutils.widgets.text("uf", "MA", "UF do recorte")

CATALOGO = dbutils.widgets.get("catalogo").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ANO = int(dbutils.widgets.get("ano"))
UF = dbutils.widgets.get("uf").strip().upper()

from pyspark.sql import functions as F
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

spark.table(f"{CATALOGO}.{SCHEMA}.silver_focos_inpe").filter(F.col("ano") == ANO).createOrReplaceTempView("focos")
spark.table("focos").filter(F.col("uf") == UF).createOrReplaceTempView("ma")
print(f"Linhas {ANO} (Brasil): {spark.table('focos').count():,} | {UF}: {spark.table('ma').count():,}")
display(spark.table(f"{CATALOGO}.{SCHEMA}.silver_focos_inpe_dicionario"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 0 — Contexto e decisor
# MAGIC
# MAGIC **Decisor:** a Secretaria de Meio Ambiente do Maranhão (SEMA-MA), junto com o Corpo de Bombeiros Militar do MA,
# MAGIC planejando a **campanha de prevenção e combate ao fogo de 2026**.
# MAGIC
# MAGIC **Decisão que depende da resposta:** usar **uma única janela de operação** para o estado inteiro
# MAGIC ou **duas janelas escalonadas**, uma para o Cerrado (sul/leste) e outra para a Amazônia maranhense (oeste).
# MAGIC Também decidir **onde** pré-posicionar brigadas e se o número de **dias sem chuva** serve como gatilho de alerta.
# MAGIC
# MAGIC **Por que o MA:** em 2025 foi a UF com mais focos do país no satélite de referência (24,3 mil) e é
# MAGIC a única grande UF dividida entre Cerrado (~71%) e Amazônia (~27%), o que permite comparar biomas
# MAGIC **sob o mesmo governo, o mesmo fuso e os mesmos satélites**.
# MAGIC
# MAGIC **Recorte escolhido:** UF = MA · ano 2025 · biomas Cerrado e Amazônia · satélite de referência AQUA_M-T
# MAGIC (outros satélites só nas checagens de instrumento).
# MAGIC
# MAGIC ## Passo 1 — O que é uma linha?
# MAGIC
# MAGIC 1. **Uma linha representa** uma detecção de foco de calor por um satélite em uma passagem. Não é um incêndio:
# MAGIC    o mesmo fogo gera várias linhas (vários satélites, várias passagens, vários dias).
# MAGIC 2. **Para contar "quanto fogo houve" vou usar** `satelite_referencia` (AQUA_M-T). Ele passa em horário fixo todos os dias
# MAGIC    do ano, então a contagem é comparável entre meses e biomas. Somar todos os satélites mede também quantos satélites
# MAGIC    estavam olhando (no MA, o GOES-19 sozinho tem ~6× mais linhas que o AQUA_M-T).
# MAGIC 3. **Medidas que posso usar:** `frp_mw` (só não existe para METOP/MSG; comparar só dentro do mesmo satélite),
# MAGIC    `numero_dias_sem_chuva` e `risco_fogo` (com nulos onde havia `-999`). **Não posso:** área queimada, área do
# MAGIC    município (sem denominador), causa do fogo, desmatamento. **Fuso:** o MA está de fato em UTC−3, então a
# MAGIC    simplificação de fuso da Silver **não** afeta este recorte.

# COMMAND ----------

# Passo 1 — satélites no recorte (quem domina a contagem? quem não tem FRP?)
display(spark.sql("""
    SELECT familia_satelite, satelite, COUNT(*) AS focos,
           ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_do_total,
           ROUND(100 * AVG(CASE WHEN frp_mw IS NULL THEN 1 ELSE 0 END), 1) AS pct_frp_nulo,
           COUNT(DISTINCT hora_local) AS horas_do_dia_com_deteccao
    FROM ma
    GROUP BY familia_satelite, satelite ORDER BY focos DESC
"""))

# COMMAND ----------

# Passo 1 — cobertura mensal: todos os satélites x referência
display(spark.sql("""
    SELECT mes, COUNT(*) AS todos, SUM(CASE WHEN satelite_referencia THEN 1 ELSE 0 END) AS referencia,
           ROUND(COUNT(*) / SUM(CASE WHEN satelite_referencia THEN 1 ELSE 0 END), 1) AS razao_todos_ref,
           COUNT(DISTINCT data) AS dias_com_dado
    FROM ma GROUP BY mes ORDER BY mes
"""))

# COMMAND ----------

# Passo 1 — biomas e nulos das medidas no recorte (referência)
display(spark.sql("""
    SELECT bioma, COUNT(*) AS focos_ref,
           ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct,
           ROUND(100 * AVG(CASE WHEN numero_dias_sem_chuva IS NULL THEN 1 ELSE 0 END), 1) AS pct_dias_sem_chuva_nulo,
           ROUND(100 * AVG(CASE WHEN risco_fogo IS NULL THEN 1 ELSE 0 END), 1) AS pct_risco_nulo,
           ROUND(100 * AVG(CASE WHEN coord_valida THEN 0 ELSE 1 END), 2) AS pct_coord_invalida
    FROM ma WHERE satelite_referencia
    GROUP BY bioma ORDER BY focos_ref DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 2 — Pré-registro
# MAGIC
# MAGIC | # | Tipo | Hipótese (com direção) | Previsão mensurável | Recorte e filtro | Métrica e gráfico | Critério de veredito | Armadilha e neutralização |
# MAGIC | --- | --- | --- | --- | --- | --- | --- | --- |
# MAGIC | H1 | Comparativa | O fogo na Amazônia maranhense acontece **mais tarde** no ano que no Cerrado maranhense | O mês mediano (mês em que se atinge 50% dos focos do ano) da Amazônia é pelo menos 1 mês depois do Cerrado | MA, 2025, só AQUA_M-T, biomas Cerrado e Amazônia | % dos focos do bioma por mês; mês mediano e centro de massa; **linhas por mês, uma por bioma** | Confirma se diferença ≥ 1 mês; refuta se ≤ 0 | O Cerrado tem ~2,6× mais focos → comparar **% dentro do bioma**, não contagens. Misturar satélites → só AQUA_M-T. Comparar com a versão "ingênua" (contagem, todos os satélites) |
# MAGIC | H2 | Espacial | O fogo no MA é concentrado: poucos municípios respondem por boa parte dos focos | Top-10 municípios (de ~216 com foco) ≥ 40% dos focos de referência do MA | MA, 2025, só AQUA_M-T | % acumulado por município; **Pareto** (barras ordenadas + curva acumulada) | Confirma ≥ 40%; refuta < 25%; inconclusiva entre | (a) Sem área: município grande ≠ fogo denso → declarar. (b) Um único incêndio grande gera muitas linhas → refazer o ranking por **dias com foco** e medir a sobreposição do top-10 |
# MAGIC | H3 | Associativa | **Dentro do Cerrado**, mais dias sem chuva acompanham fogo mais intenso | FRP mediana cresce de forma monotônica por faixa de dias sem chuva, e a faixa "31+" é ≥ 1,3× a faixa "0-2" | MA, Cerrado, um satélite por vez (AQUA_M-T e NOAA-20), FRP não nulo | Mediana de FRP por faixa; **barras agrupadas por satélite** | Confirma se monotônica e razão ≥ 1,3 nos dois; refuta se a razão < 1,0 nos dois; inconclusiva nos demais casos | Misturar sensores; média puxada por extremos → mediana e um satélite por vez. **Confusão com a estação:** seca e fim do ano andam juntos → repetir só em jul–out |
# MAGIC | H4 | Instrumento | O atraso da Amazônia (H1) é do **fogo**, não do satélite | A diferença de mês mediano Amazônia − Cerrado é ≥ 1 em AQUA_M-T, NOAA-20 **e** GOES-19 | MA, 2025, Cerrado e Amazônia, cada satélite separado | Mês mediano por bioma e satélite; **linhas por mês, um painel por satélite** | Confirma se ≥ 1 nos 3 satélites; refuta se em nenhum; inconclusiva se em parte | A razão todos/referência pode variar por bioma (cobertura, nuvem) → mostrar a razão por bioma e por mês |
# MAGIC | H5 | Temporal | O fogo no MA é atividade humana de manejo: **fins de semana têm menos focos** que dias úteis | Média de focos por dia em sáb+dom ≤ 0,90 × a média de seg–sex | MA, 2025, só AQUA_M-T, **todos os dias do ano** (dias sem foco contam como 0) | Média de focos/dia por dia da semana; **barras por dia da semana** | Confirma se razão ≤ 0,90; refuta se ≥ 0,97; inconclusiva entre | Dias sem foco somem do GROUP BY → usar um calendário completo. A sazonalidade domina a variância → repetir só na estação de fogo (jul–nov) e com o GOES-19. Fogo que começa na sexta é detectado no sábado → declarar |
# MAGIC
# MAGIC **Ordem de execução:** H1 → H4 → H2 → H3 → H5.
# MAGIC H1 responde a decisão principal (uma janela ou duas) e é barata. H4 vem logo depois porque valida H1: se o atraso
# MAGIC for do satélite, a recomendação muda. H2 define onde posicionar as brigadas. H3 e H5 refinam os gatilhos operacionais.
# MAGIC
# MAGIC **Declaração de transparência:** antes do pré-registro, a exploração do Passo 1 (feita no SQL Editor) já mostrou
# MAGIC os focos de referência do MA **por mês e por bioma**. A H1 foi formulada depois de ver essa tabela.
# MAGIC Por isso, a H4 (outros satélites) é a checagem independente da H1. H2, H3 e H5 não foram consultadas antes.

# COMMAND ----------

# Função auxiliar: registre o veredito de cada hipótese aqui para a síntese final
VEREDITOS = {}

def registrar(codigo, metrica_observada, criterio, veredito, armadilha_verificada, limitacao=""):
    VEREDITOS[codigo] = dict(metrica=metrica_observada, criterio=criterio, veredito=veredito,
                             armadilha=armadilha_verificada, limitacao=limitacao)
    print(f"{codigo}: {metrica_observada} | critério: {criterio} → {veredito}")


BIOMAS = ["Cerrado", "Amazônia"]


def calendario(filtro_sql: str) -> pd.DataFrame:
    """Focos por mês (linhas 1..12) e bioma (colunas Cerrado, Amazônia) no MA, com o filtro dado."""
    p = spark.sql(f"""
        SELECT bioma, mes, COUNT(*) AS focos
        FROM ma WHERE ({filtro_sql}) AND bioma IN ('Cerrado', 'Amazônia')
        GROUP BY bioma, mes
    """).toPandas()
    t = p.pivot(index="mes", columns="bioma", values="focos").reindex(range(1, 13)).fillna(0)
    t = t.reindex(columns=BIOMAS, fill_value=0)
    t.columns.name = None
    t.index.name = "mes"
    return t


def mes_mediano(serie: pd.Series) -> int:
    """Primeiro mês em que o acumulado do ano atinge 50% dos focos."""
    acum = serie.cumsum() / serie.sum()
    return int(acum[acum >= 0.5].index[0])


def centro_de_massa(serie: pd.Series) -> float:
    """Mês médio ponderado pelos focos (1 = jan ... 12 = dez)."""
    return float((serie.index.to_series() * serie).sum() / serie.sum())


def em_percentual(t: pd.DataFrame) -> pd.DataFrame:
    return (100 * t / t.sum()).round(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## H1 · Comparativa — "o fogo na Amazônia maranhense acontece mais tarde que no Cerrado"
# MAGIC **Gráfico escolhido e por quê:** linhas por mês, uma por bioma, em **% do total do bioma**. É uma comparação de
# MAGIC *forma* do calendário, não de volume, por isso as duas curvas precisam estar na mesma escala.

# COMMAND ----------

# H1 — consulta principal: calendário por bioma, só satélite de referência
cal_ref = calendario("satelite_referencia")
pct_ref = em_percentual(cal_ref)
display(cal_ref.astype(int).join(pct_ref, rsuffix="_pct").reset_index())

fig = px.line(pct_ref.reset_index().melt(id_vars="mes", var_name="bioma", value_name="pct_do_bioma"),
              x="mes", y="pct_do_bioma", color="bioma", markers=True, height=420,
              title=f"H1 — % dos focos do bioma por mês, {UF} {ANO} (AQUA_M-T)")
fig.update_xaxes(dtick=1)
fig.show()

# COMMAND ----------

# H1 — veredito contra o critério pré-registrado
med = {b: mes_mediano(cal_ref[b]) for b in BIOMAS}
cm = {b: centro_de_massa(cal_ref[b]) for b in BIOMAS}
dif_h1 = med["Amazônia"] - med["Cerrado"]
print(f"Mês mediano — Cerrado: {med['Cerrado']} | Amazônia: {med['Amazônia']} | diferença: {dif_h1:+d} mês(es)")
print(f"Centro de massa — Cerrado: {cm['Cerrado']:.2f} | Amazônia: {cm['Amazônia']:.2f} | diferença: {cm['Amazônia'] - cm['Cerrado']:+.2f}")
print(f"n (AQUA_M-T) — Cerrado: {int(cal_ref['Cerrado'].sum()):,} | Amazônia: {int(cal_ref['Amazônia'].sum()):,}")

veredito_h1 = "CONFIRMADA" if dif_h1 >= 1 else "REFUTADA"
registrar("H1", f"mês mediano Amazônia − Cerrado = {dif_h1:+d}", "confirma ≥ 1; refuta ≤ 0", veredito_h1,
          armadilha_verificada="comparação em % do bioma, só AQUA_M-T; versão ingênua calculada abaixo",
          limitacao="um ano só; bioma é do pixel, não do município")

# COMMAND ----------

# H1 — a armadilha: o mesmo cálculo em contagem absoluta, somando todos os satélites
cal_todos = calendario("TRUE")
med_todos = {b: mes_mediano(cal_todos[b]) for b in BIOMAS}
print(f"Todos os satélites — mês mediano Cerrado: {med_todos['Cerrado']} | Amazônia: {med_todos['Amazônia']} "
      f"| diferença: {med_todos['Amazônia'] - med_todos['Cerrado']:+d}")
fig = px.line(cal_todos.reset_index().melt(id_vars="mes", var_name="bioma", value_name="focos"),
              x="mes", y="focos", color="bioma", markers=True, height=380,
              title="H1 (ingênuo) — contagem absoluta, todos os satélites: o Cerrado 'engole' a curva da Amazônia")
fig.update_xaxes(dtick=1)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H1 (3 linhas):** _o que o número diz (diferença de __ meses), o que ele não diz (não explica **por quê**:
# MAGIC calendário de chuvas diferente? tipo de uso da terra?) e que hipótese nova surgiu._
# MAGIC
# MAGIC ---
# MAGIC ## H4 · Instrumento — "o atraso da Amazônia é do fogo, não do satélite"
# MAGIC **Gráfico escolhido e por quê:** as mesmas linhas da H1, um painel por satélite. Se as curvas mudarem de forma entre
# MAGIC os painéis, o padrão é do instrumento.

# COMMAND ----------

# H4 — consulta principal: o calendário por bioma em 3 satélites independentes
SATS_H4 = ["AQUA_M-T", "NOAA-20", "GOES-19"]
linhas, resumo = [], []
for sat in SATS_H4:
    c = calendario(f"satelite = '{sat}'")
    p = em_percentual(c).reset_index().melt(id_vars="mes", var_name="bioma", value_name="pct_do_bioma")
    p["satelite"] = sat
    linhas.append(p)
    m = {b: mes_mediano(c[b]) for b in BIOMAS}
    resumo.append(dict(satelite=sat, n_cerrado=int(c["Cerrado"].sum()), n_amazonia=int(c["Amazônia"].sum()),
                       mes_mediano_cerrado=m["Cerrado"], mes_mediano_amazonia=m["Amazônia"],
                       diferenca=m["Amazônia"] - m["Cerrado"]))
h4 = pd.DataFrame(resumo)
display(h4)

fig = px.line(pd.concat(linhas), x="mes", y="pct_do_bioma", color="bioma", facet_col="satelite",
              markers=True, height=400, title="H4 — % dos focos do bioma por mês, por satélite")
fig.update_xaxes(dtick=2)
fig.show()

# COMMAND ----------

# H4 — veredito
n_ok = int((h4["diferenca"] >= 1).sum())
veredito_h4 = "CONFIRMADA" if n_ok == len(SATS_H4) else ("REFUTADA" if n_ok == 0 else "INCONCLUSIVA")
registrar("H4", f"atraso ≥ 1 mês em {n_ok} de {len(SATS_H4)} satélites", "confirma 3/3; refuta 0/3", veredito_h4,
          armadilha_verificada="razão todos/referência por bioma e mês (célula abaixo)",
          limitacao="satélites polares só veem 1–2 horários por dia")

# COMMAND ----------

# H4 — a armadilha: o "peso" dos outros satélites é igual nos dois biomas e ao longo do ano?
display(spark.sql("""
    SELECT bioma, mes, COUNT(*) AS todos,
           SUM(CASE WHEN satelite_referencia THEN 1 ELSE 0 END) AS referencia,
           ROUND(COUNT(*) / NULLIF(SUM(CASE WHEN satelite_referencia THEN 1 ELSE 0 END), 0), 1) AS razao_todos_ref
    FROM ma WHERE bioma IN ('Cerrado', 'Amazônia')
    GROUP BY bioma, mes ORDER BY bioma, mes
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H4 (3 linhas):** _o atraso aparece nos três satélites? Se a razão todos/referência muda muito entre
# MAGIC biomas, qualquer comparação "todos os satélites" entre Cerrado e Amazônia estaria enviesada._
# MAGIC
# MAGIC ---
# MAGIC ## H2 · Espacial — "poucos municípios concentram boa parte dos focos do MA"
# MAGIC **Gráfico escolhido e por quê:** Pareto (barras ordenadas + curva acumulada). A pergunta é sobre concentração,
# MAGIC e o Pareto mostra direto quantos municípios são necessários para chegar a X%.

# COMMAND ----------

# H2 — consulta principal: Pareto dos municípios (só referência), com dias distintos com foco para a armadilha
h2 = spark.sql("""
    WITH m AS (
        SELECT cod_ibge_municipio, municipio, COUNT(*) AS focos, COUNT(DISTINCT data) AS dias_com_foco
        FROM ma WHERE satelite_referencia
        GROUP BY cod_ibge_municipio, municipio
    ),
    r AS (SELECT *, ROW_NUMBER() OVER (ORDER BY focos DESC) AS posicao, SUM(focos) OVER () AS total FROM m)
    SELECT posicao, municipio, cod_ibge_municipio, focos, dias_com_foco,
           ROUND(100 * focos / total, 1) AS pct,
           ROUND(100 * SUM(focos) OVER (ORDER BY posicao) / total, 1) AS pct_acumulado
    FROM r ORDER BY posicao
""").toPandas()
display(h2.head(20))

top = h2.head(30)
fig = go.Figure()
fig.add_bar(x=top["municipio"], y=top["focos"], name="focos (AQUA_M-T)")
fig.add_scatter(x=top["municipio"], y=top["pct_acumulado"], name="% acumulado", yaxis="y2", mode="lines+markers")
fig.update_layout(title=f"H2 — Pareto dos 30 municípios com mais focos, {UF} {ANO}", height=450,
                  yaxis2=dict(overlaying="y", side="right", range=[0, 100], title="% acumulado"),
                  yaxis=dict(title="focos"), xaxis=dict(tickangle=-45))
fig.show()

# COMMAND ----------

# H2 — veredito
n_munis = len(h2)
pct_top10 = float(h2.loc[h2["posicao"] == 10, "pct_acumulado"].iloc[0])
veredito_h2 = "CONFIRMADA" if pct_top10 >= 40 else ("REFUTADA" if pct_top10 < 25 else "INCONCLUSIVA")
registrar("H2", f"top-10 de {n_munis} municípios = {pct_top10:.1f}%", "confirma ≥ 40%; refuta < 25%", veredito_h2,
          armadilha_verificada="ranking por dias com foco (célula abaixo)",
          limitacao="sem área municipal: concentração absoluta, não densidade")

# COMMAND ----------

# H2 — a armadilha: o ranking muda se contarmos DIAS com foco em vez de linhas?
top10_focos = set(h2.nsmallest(10, "posicao")["municipio"])
top10_dias = set(h2.nlargest(10, "dias_com_foco")["municipio"])
comum = top10_focos & top10_dias
print(f"Sobreposição top-10 (focos) × top-10 (dias com foco): {len(comum)} de 10")
print("Só no ranking por focos:", sorted(top10_focos - top10_dias))
print("Só no ranking por dias  :", sorted(top10_dias - top10_focos))
display(h2.assign(focos_por_dia=(h2["focos"] / h2["dias_com_foco"]).round(1)).head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H2 (3 linhas):** _quais municípios lideram e em que bioma ficam? Se a sobreposição for alta, a concentração
# MAGIC não é efeito de um único incêndio enorme. Lembrar: municípios como Balsas e Grajaú são enormes, então "mais focos"
# MAGIC não é "mais fogo por km²"._
# MAGIC
# MAGIC ---
# MAGIC ## H3 · Associativa — "dentro do Cerrado, mais dias sem chuva acompanham fogo mais intenso"
# MAGIC **Gráfico escolhido e por quê:** barras da mediana de FRP por faixa de dias sem chuva, agrupadas por satélite.
# MAGIC Uma associação com variável categórica ordenada pede medianas por faixa. FRP não se compara entre sensores.

# COMMAND ----------

# H3 — consulta principal: mediana de FRP por faixa, Cerrado do MA, por satélite
ORDEM_FAIXAS = ["0-2 dias", "3-7 dias", "8-15 dias", "16-30 dias", "31+ dias"]
SATS_H3 = ["AQUA_M-T", "NOAA-20"]


def frp_por_faixa(filtro_extra: str = "TRUE") -> pd.DataFrame:
    p = spark.sql(f"""
        SELECT satelite, faixa_dias_sem_chuva, COUNT(*) AS n,
               ROUND(percentile_approx(frp_mw, 0.5), 1) AS frp_mediana_mw,
               ROUND(AVG(frp_mw), 1) AS frp_media_mw
        FROM ma
        WHERE bioma = 'Cerrado' AND frp_mw IS NOT NULL AND faixa_dias_sem_chuva <> 'sem info'
          AND satelite IN ('AQUA_M-T', 'NOAA-20') AND ({filtro_extra})
        GROUP BY satelite, faixa_dias_sem_chuva
    """).toPandas()
    p["faixa_dias_sem_chuva"] = pd.Categorical(p["faixa_dias_sem_chuva"], categories=ORDEM_FAIXAS, ordered=True)
    return p.sort_values(["satelite", "faixa_dias_sem_chuva"])


def avaliar_h3(p: pd.DataFrame) -> dict:
    res = {}
    for sat in SATS_H3:
        g = p[p["satelite"] == sat].set_index("faixa_dias_sem_chuva")
        serie = g["frp_mediana_mw"].reindex(ORDEM_FAIXAS)
        n_min = int(g["n"].reindex(ORDEM_FAIXAS).fillna(0).min())
        razao = serie["31+ dias"] / serie["0-2 dias"] if serie["0-2 dias"] else float("nan")
        res[sat] = dict(monotonica=bool(serie.is_monotonic_increasing and serie.notna().all()),
                        razao=float(razao), n_min=n_min)
        print(f"  {sat}: monotônica={res[sat]['monotonica']}, razão 31+/0-2 = {razao:.2f}, menor n por faixa = {n_min}")
    return res


h3 = frp_por_faixa()
display(h3)
fig = px.bar(h3, x="faixa_dias_sem_chuva", y="frp_mediana_mw", color="satelite", barmode="group", height=420,
             title=f"H3 — FRP mediana por faixa de dias sem chuva, Cerrado do {UF} {ANO}")
fig.show()

# COMMAND ----------

# H3 — veredito contra o critério pré-registrado
r3 = avaliar_h3(h3)
ok = [v["monotonica"] and v["razao"] >= 1.3 for v in r3.values()]
if all(ok):
    veredito_h3 = "CONFIRMADA"
elif all(v["razao"] < 1.0 for v in r3.values()):
    veredito_h3 = "REFUTADA"
else:
    veredito_h3 = "INCONCLUSIVA"
registrar("H3", "; ".join(f"{s}: razão {v['razao']:.2f}, monot.={v['monotonica']}" for s, v in r3.items()),
          "confirma monotônica e ≥ 1,3 nos 2; refuta < 1,0 nos 2", veredito_h3,
          armadilha_verificada="repetido só na estação seca (jul–out), célula abaixo",
          limitacao="associação, não causa; dias sem chuva é estimado pelo INPE no pixel")

# COMMAND ----------

# H3 — a armadilha: a relação sobrevive dentro da estação seca (jul–out)? Se sumir, era efeito da estação.
h3_seca = frp_por_faixa("mes BETWEEN 7 AND 10")
display(h3_seca)
print("Só jul–out:")
r3_seca = avaliar_h3(h3_seca)

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H3 (3 linhas):** _a razão se mantém dentro da estação seca? Se sim, "dias sem chuva" é um gatilho
# MAGIC operacional razoável dentro da campanha; se não, o que parecia efeito da seca era só o calendário._
# MAGIC
# MAGIC ---
# MAGIC ## H5 · Temporal — "fins de semana têm menos focos que dias úteis"
# MAGIC **Gráfico escolhido e por quê:** barras da média de focos por dia, por dia da semana. Comparar médias por dia (não
# MAGIC somas) corrige o fato de o ano ter 52 ou 53 de cada dia da semana.

# COMMAND ----------

# H5 — consulta principal: calendário completo (dias sem foco = 0), só referência
def focos_por_dia(filtro_sat: str) -> pd.DataFrame:
    return spark.sql(f"""
        WITH cal AS (SELECT explode(sequence(to_date('{ANO}-01-01'), to_date('{ANO}-12-31'), interval 1 day)) AS data),
             d AS (SELECT data, COUNT(*) AS focos FROM ma WHERE {filtro_sat} GROUP BY data)
        SELECT c.data, month(c.data) AS mes, dayofweek(c.data) AS dow, COALESCE(d.focos, 0) AS focos
        FROM cal c LEFT JOIN d ON c.data = d.data
    """).toPandas()


NOMES_DOW = {1: "dom", 2: "seg", 3: "ter", 4: "qua", 5: "qui", 6: "sex", 7: "sáb"}


def razao_fds(dias: pd.DataFrame, estat: str = "mean") -> float:
    fds = dias["dow"].isin([1, 7])
    return float(dias.loc[fds, "focos"].agg(estat) / dias.loc[~fds, "focos"].agg(estat))


dias_ref = focos_por_dia("satelite_referencia")
por_dow = (dias_ref.groupby("dow")["focos"].agg(["mean", "median", "count"]).round(1)
           .reset_index().assign(dia=lambda d: d["dow"].map(NOMES_DOW)))
display(por_dow)
fig = px.bar(por_dow, x="dia", y="mean", height=380, title=f"H5 — média de focos/dia por dia da semana, {UF} {ANO} (AQUA_M-T)")
fig.show()

# COMMAND ----------

# H5 — veredito
razao_h5 = razao_fds(dias_ref)
veredito_h5 = "CONFIRMADA" if razao_h5 <= 0.90 else ("REFUTADA" if razao_h5 >= 0.97 else "INCONCLUSIVA")
registrar("H5", f"média fds / média seg–sex = {razao_h5:.2f}", "confirma ≤ 0,90; refuta ≥ 0,97", veredito_h5,
          armadilha_verificada="repetido em jul–nov, com mediana e com GOES-19 (célula abaixo)",
          limitacao="fogo dura vários dias; detecção ≠ ignição")

# COMMAND ----------

# H5 — a armadilha: sazonalidade e satélite. A razão se mantém na estação de fogo e em outro sensor?
dias_goes = focos_por_dia("satelite = 'GOES-19'")
checagens = pd.DataFrame([
    dict(versao="AQUA_M-T, ano todo, média", razao=razao_fds(dias_ref)),
    dict(versao="AQUA_M-T, jul–nov, média", razao=razao_fds(dias_ref[dias_ref["mes"].between(7, 11)])),
    dict(versao="AQUA_M-T, jul–nov, mediana", razao=razao_fds(dias_ref[dias_ref["mes"].between(7, 11)], "median")),
    dict(versao="GOES-19, jul–nov, média", razao=razao_fds(dias_goes[dias_goes["mes"].between(7, 11)])),
]).round(3)
display(checagens)

# COMMAND ----------

# MAGIC %md
# MAGIC **Discussão H5 (3 linhas):** _se for refutada, é um resultado útil: o fogo detectado não segue o ritmo da semana
# MAGIC (queimas duram dias, ou o manejo não para no fim de semana), e a escala das brigadas não pode relaxar no sábado._
# MAGIC
# MAGIC ---
# MAGIC ## Passo final — Síntese para o decisor (SEMA-MA / CBMMA)
# MAGIC
# MAGIC Preencha com os números impressos pelo `registrar` (tabela automática logo abaixo).
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Achado 1** | H1+H4: o mês mediano do fogo na Amazônia maranhense é __ meses depois do Cerrado, em __ de 3 satélites → **duas janelas de operação** (Cerrado ~__–__, Amazônia ~__–__) |
# MAGIC | **Achado 2** | H2: os 10 municípios do topo concentram __% dos focos do MA (__ de 10 se mantêm no ranking por dias com foco) → lista nominal para pré-posicionar brigadas |
# MAGIC | **Achado 3** | H3: no Cerrado, FRP mediana na faixa 31+ dias sem chuva é __× a da faixa 0–2 (__ dentro da estação seca) → dias sem chuva como gatilho de alerta? |
# MAGIC | **Limitação 1** | Foco ≠ incêndio; séries só com AQUA_M-T (1 passagem diurna), que subamostra o fogo |
# MAGIC | **Limitação 2** | Um único ano (2025); sem área municipal (concentração ≠ densidade); MA inteiro marcado como Amazônia Legal na Silver |
# MAGIC | **Limitação 3** | Associação não é causa: seca, estação e uso da terra andam juntos; bioma é do pixel |
# MAGIC | **O que NÃO encontramos (e esperávamos)** | H5: __ (efeito fim de semana) |
# MAGIC | **Hipóteses novas que surgiram** | _ex.: o atraso da Amazônia acompanha o atraso do fim das chuvas no oeste do MA? O pico de nov/dez é queima de roça para o plantio?_ |
# MAGIC | **Próximo passo (Gold / outra fonte)** | `gold_focos_mes_bioma_uf` (só referência); juntar área IBGE por `cod_ibge_municipio`; precipitação mensal (INMET/CHIRPS); repetir com 2023–2024 |
# MAGIC
# MAGIC **Mudanças de critério depois de ver os dados:** nenhuma no critério. A H1 foi formulada depois de ver a tabela
# MAGIC mês × bioma no Passo 1 (declarado no Passo 2); a H4 é a checagem independente.

# COMMAND ----------

# Resumo automático dos vereditos registrados
display(pd.DataFrame.from_dict(VEREDITOS, orient="index").reset_index().rename(columns={"index": "hipotese"}))

# COMMAND ----------

# (Opcional) Gold nascida da H1/H4: calendário mensal por bioma e UF, só satélite de referência
spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOGO}.{SCHEMA}.gold_focos_mes_bioma_uf AS
SELECT ano, mes, uf, bioma, COUNT(*) AS focos_referencia
FROM {CATALOGO}.{SCHEMA}.silver_focos_inpe
WHERE satelite_referencia
GROUP BY ano, mes, uf, bioma
""")
display(spark.table(f"{CATALOGO}.{SCHEMA}.gold_focos_mes_bioma_uf").filter(F.col("uf") == UF).orderBy("bioma", "mes"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checklist de entrega
# MAGIC - [ ] Passos 0, 1 e 2 preenchidos antes das consultas (validado pelo professor)
# MAGIC - [ ] Cada hipótese tem: consulta, gráfico adequado, veredito numérico, armadilha checada, discussão
# MAGIC - [ ] Pelo menos uma hipótese é de **instrumento** (H4) ou verificou o efeito do satélite (H1, H5)
# MAGIC - [ ] Síntese com 3 achados + 3 limitações + próximo passo
# MAGIC - [ ] Notebook executado do início ao fim sem erro (Run all)