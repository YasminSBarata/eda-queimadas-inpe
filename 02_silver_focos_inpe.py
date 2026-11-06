# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 02 · Silver — Focos de calor do INPE, limpos e enriquecidos
# MAGIC
# MAGIC **Objetivo desta camada:** transformar o texto bruto da Bronze em uma tabela **tipada, consistente e
# MAGIC documentada**, pronta para a análise exploratória. A Silver responde às perguntas que a Bronze levantou:
# MAGIC
# MAGIC | Defeito visto na Bronze | Decisão na Silver |
# MAGIC | --- | --- |
# MAGIC | `lat`/`lon` com espaço antes do sinal | `trim` + `cast double`; coordenadas fora do Brasil viram `coord_valida = false` |
# MAGIC | `-999` em `numero_dias_sem_chuva`, `precipitacao`, `risco_fogo` | Vira `NULL` (é "sem informação", não é um número) |
# MAGIC | `risco_fogo` fora de [0, 1] | `NULL` |
# MAGIC | `frp` vazio (METOP) | `NULL` — e a coluna `familia_satelite` lembra que FRP só se compara dentro do mesmo sensor |
# MAGIC | Horário em UTC | Mantém `data_hora_utc` **e** cria `data_hora_local` (fuso `America/Belem`, UTC−3) |
# MAGIC | 10+ satélites misturados | Mantém todos, mas marca `satelite_referencia = (satelite == 'AQUA_M-T')` |
# MAGIC | `id` repetido entre arquivos | Remove duplicatas exatas por `id` |
# MAGIC
# MAGIC A Silver **não agrega e não filtra por satélite** — ela só torna o filtro fácil. Agregar é papel da EDA/Gold.
# MAGIC
# MAGIC > **Simplificação declarada:** usamos um único fuso (UTC−3) para todo o Brasil. Acre, Amazonas (oeste),
# MAGIC > Roraima, Rondônia, MS e MT estão em UTC−4/−5. Para hipóteses sobre *hora do dia* nesses estados,
# MAGIC > a hora local pode estar deslocada em 1–2 h. Isso é uma limitação a registrar, não a esconder.

# COMMAND ----------

dbutils.widgets.text("catalogo", "workspace", "Catálogo")
dbutils.widgets.text("schema", "queimadas", "Schema")

CATALOGO = dbutils.widgets.get("catalogo").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
TABELA_BRONZE = f"{CATALOGO}.{SCHEMA}.bronze_focos_inpe"
TABELA_SILVER = f"{CATALOGO}.{SCHEMA}.silver_focos_inpe"
TABELA_DICIONARIO = f"{CATALOGO}.{SCHEMA}.silver_focos_inpe_dicionario"
FUSO_LOCAL = "America/Belem"

print("Lendo  :", TABELA_BRONZE)
print("Gravando:", TABELA_SILVER)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Regras de transformação
# MAGIC
# MAGIC Tudo está em **uma função pura** (`transformar_silver`) que recebe o DataFrame Bronze e devolve o Silver.
# MAGIC Vantagens: dá para testar com um CSV pequeno, dá para ler as regras de cima a baixo, e a Silver fica reprodutível.

# COMMAND ----------

# --- INICIO transformacao ---
from pyspark.sql import DataFrame, functions as F, types as T

# Código IBGE da UF -> sigla. O primeiro dígito do código é a região (1 Norte ... 5 Centro-Oeste).
UF_POR_CODIGO = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA",
    31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS",
    50: "MS", 51: "MT", 52: "GO", 53: "DF",
}
REGIAO_POR_DIGITO = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}
# dayofweek() do Spark: 1 = domingo ... 7 = sábado (independe do locale do cluster).
DIA_SEMANA = {1: "domingo", 2: "segunda", 3: "terça", 4: "quarta", 5: "quinta", 6: "sexta", 7: "sábado"}

# Amazônia Legal: 8 estados inteiros + Maranhão (oficialmente só a oeste do meridiano 44°W;
# aqui incluímos o MA inteiro — simplificação declarada).
UFS_AMAZONIA_LEGAL = {"AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO"}

SATELITE_REFERENCIA = "AQUA_M-T"

# Caixa que contém o Brasil continental (folga de ~1 grau). Fora dela, a coordenada é suspeita.
LAT_MIN, LAT_MAX = -34.5, 6.0
LON_MIN, LON_MAX = -74.5, -33.5


def _num(col: str, tipo):
    """Texto -> número. Espaços são removidos; vazio vira NULL; cast inválido vira NULL (comportamento do Spark)."""
    c = F.trim(F.col(col))
    return F.when(c == "", None).otherwise(c).cast(tipo)


def _sem_sentinela(expr, minimo=None, maximo=None):
    """Troca a sentinela -999 (e qualquer valor fora da faixa válida) por NULL."""
    cond = expr != -999
    if minimo is not None:
        cond = cond & (expr >= minimo)
    if maximo is not None:
        cond = cond & (expr <= maximo)
    return F.when(cond, expr).otherwise(None)


def transformar_silver(bronze: DataFrame, fuso_local: str = "America/Belem") -> DataFrame:
    """Aplica tipagem, limpeza de sentinelas, enriquecimento geográfico/temporal e deduplicação por id."""
    uf_map = F.create_map(*[F.lit(x) for kv in UF_POR_CODIGO.items() for x in kv])
    regiao_map = F.create_map(*[F.lit(x) for kv in REGIAO_POR_DIGITO.items() for x in kv])
    dia_semana_map = F.create_map(*[F.lit(x) for kv in DIA_SEMANA.items() for x in kv])

    df = (
        bronze
        # ---- identificação ----
        .withColumn("id", F.trim(F.col("id")))
        # ---- coordenadas ----
        .withColumn("lat", _num("lat", T.DoubleType()))
        .withColumn("lon", _num("lon", T.DoubleType()))
        .withColumn(
            "coord_valida",
            F.col("lat").isNotNull() & F.col("lon").isNotNull()
            & F.col("lat").between(LAT_MIN, LAT_MAX) & F.col("lon").between(LON_MIN, LON_MAX),
        )
        # ---- tempo ----
        .withColumn("data_hora_utc", F.to_timestamp(F.trim(F.col("data_hora_gmt")), "yyyy-MM-dd HH:mm:ss"))
        .withColumn("data_hora_local", F.from_utc_timestamp(F.col("data_hora_utc"), fuso_local))
        .withColumn("data", F.to_date("data_hora_local"))
        .withColumn("ano", F.year("data_hora_local"))
        .withColumn("mes", F.month("data_hora_local"))
        .withColumn("dia", F.dayofmonth("data_hora_local"))
        .withColumn("hora_local", F.hour("data_hora_local"))
        .withColumn("dia_semana", dia_semana_map[F.dayofweek("data_hora_local")])
        # ---- satélite ----
        .withColumn("satelite", F.trim(F.col("satelite")))
        .withColumn("satelite_referencia", F.col("satelite") == SATELITE_REFERENCIA)
        .withColumn("familia_satelite", F.regexp_extract(F.col("satelite"), r"^([A-Za-z]+)", 1))
        # ---- geografia ----
        .withColumn("municipio", F.trim(F.col("municipio")))
        .withColumn("estado", F.trim(F.col("estado")))
        .withColumn("cod_ibge_municipio", F.lpad(F.trim(F.col("municipio_id")), 7, "0"))
        .withColumn("cod_ibge_uf", _num("estado_id", T.IntegerType()))
        .withColumn("uf", uf_map[F.col("cod_ibge_uf")])
        .withColumn("regiao", regiao_map[(F.col("cod_ibge_uf") / 10).cast("int")])
        .withColumn("amazonia_legal", F.coalesce(F.col("uf").isin(sorted(UFS_AMAZONIA_LEGAL)), F.lit(False)))
        .withColumn("bioma", F.trim(F.col("bioma")))
        # ---- medidas (com tratamento de sentinela) ----
        .withColumn("numero_dias_sem_chuva", _sem_sentinela(_num("numero_dias_sem_chuva", T.DoubleType()), minimo=0))
        .withColumn("precipitacao_mm", _sem_sentinela(_num("precipitacao", T.DoubleType()), minimo=0))
        .withColumn("risco_fogo", _sem_sentinela(_num("risco_fogo", T.DoubleType()), minimo=0, maximo=1))
        .withColumn("frp_mw", _num("frp", T.DoubleType()))
        # ---- faixa categórica útil para a EDA ----
        .withColumn(
            "faixa_dias_sem_chuva",
            F.when(F.col("numero_dias_sem_chuva").isNull(), "sem info")
             .when(F.col("numero_dias_sem_chuva") <= 2, "0-2 dias")
             .when(F.col("numero_dias_sem_chuva") <= 7, "3-7 dias")
             .when(F.col("numero_dias_sem_chuva") <= 15, "8-15 dias")
             .when(F.col("numero_dias_sem_chuva") <= 30, "16-30 dias")
             .otherwise("31+ dias"),
        )
    )

    colunas = [
        "id", "data_hora_utc", "data_hora_local", "data", "ano", "mes", "dia", "hora_local", "dia_semana",
        "lat", "lon", "coord_valida",
        "satelite", "satelite_referencia", "familia_satelite",
        "pais", "regiao", "uf", "cod_ibge_uf", "estado", "cod_ibge_municipio", "municipio", "amazonia_legal", "bioma",
        "numero_dias_sem_chuva", "faixa_dias_sem_chuva", "precipitacao_mm", "risco_fogo", "frp_mw",
        "_arquivo_origem", "_ano_mes",
    ]
    # Deduplicação: o mesmo id pode aparecer em mais de um arquivo (reprocessamentos do INPE).
    return df.select(*colunas).dropDuplicates(["id"])


DICIONARIO = [
    ("id", "UUID do registro no INPE. Chave da deduplicação."),
    ("data_hora_utc", "Instante da detecção em UTC (campo original data_hora_gmt)."),
    ("data_hora_local", "data_hora_utc convertida para America/Belem (UTC-3). Simplificação: fuso único para o Brasil."),
    ("data", "Data local da detecção."),
    ("ano / mes / dia / hora_local / dia_semana", "Partes derivadas de data_hora_local."),
    ("lat / lon", "Coordenadas WGS84 do pixel. Espaços removidos e cast para double."),
    ("coord_valida", "true se lat/lon não nulos e dentro da caixa do Brasil continental."),
    ("satelite", "Nome do satélite/sensor como no INPE."),
    ("satelite_referencia", "true apenas para AQUA_M-T, o satélite de referência do INPE para séries comparáveis."),
    ("familia_satelite", "Prefixo do satélite (AQUA, TERRA, NOAA, NPP, METOP, GOES, MSG). FRP só é comparável dentro da mesma família."),
    ("regiao / uf / cod_ibge_uf / estado", "Geografia estadual derivada do código IBGE da UF."),
    ("cod_ibge_municipio / municipio", "Código IBGE (7 dígitos) e nome do município."),
    ("amazonia_legal", "true para AC, AM, AP, MA, MT, PA, RO, RR, TO (MA inteiro - simplificação)."),
    ("bioma", "Bioma do pixel segundo o INPE."),
    ("numero_dias_sem_chuva", "Dias consecutivos sem chuva. -999 e negativos viraram NULL."),
    ("faixa_dias_sem_chuva", "Faixa categórica de numero_dias_sem_chuva para gráficos."),
    ("precipitacao_mm", "Precipitação acumulada (mm). -999 e negativos viraram NULL."),
    ("risco_fogo", "Risco de fogo previsto, 0 a 1. -999 e fora da faixa viraram NULL."),
    ("frp_mw", "Fire Radiative Power (MW). Vazio (METOP) virou NULL. Escala varia por sensor."),
    ("_arquivo_origem / _ano_mes", "Proveniência: arquivo mensal do INPE de onde a linha veio."),
]
# --- FIM transformacao ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Executar e gravar

# COMMAND ----------

bronze = spark.table(TABELA_BRONZE)
silver = transformar_silver(bronze, FUSO_LOCAL)

(
    silver.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("ano", "mes")
    .saveAsTable(TABELA_SILVER)
)

spark.createDataFrame(DICIONARIO, ["coluna", "descricao"]).write.format("delta").mode("overwrite").saveAsTable(TABELA_DICIONARIO)

spark.sql(f"""
COMMENT ON TABLE {TABELA_SILVER} IS
'Focos de calor INPE (camada Silver): tipada, sem sentinelas -999, com fuso local, flags de satélite de referência e geografia IBGE. Todos os satélites mantidos; filtre por satelite_referencia para séries comparáveis.'
""")
print("Gravado:", TABELA_SILVER, "e", TABELA_DICIONARIO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reconciliação Bronze → Silver
# MAGIC
# MAGIC Toda transformação precisa prestar contas: quantas linhas entraram, quantas saíram, e por quê.

# COMMAND ----------

n_bronze = bronze.count()
s = spark.table(TABELA_SILVER)
n_silver = s.count()

print(f"Bronze: {n_bronze:,} linhas")
print(f"Silver: {n_silver:,} linhas  (removidas por id duplicado: {n_bronze - n_silver:,})")

display(
    s.select(
        F.count("*").alias("linhas"),
        F.sum(F.when(~F.col("coord_valida"), 1).otherwise(0)).alias("coord_invalida"),
        F.sum(F.when(F.col("data_hora_utc").isNull(), 1).otherwise(0)).alias("data_nula"),
        F.sum(F.when(F.col("uf").isNull(), 1).otherwise(0)).alias("uf_nula"),
        F.sum(F.when(F.col("numero_dias_sem_chuva").isNull(), 1).otherwise(0)).alias("dias_sem_chuva_nulo"),
        F.sum(F.when(F.col("risco_fogo").isNull(), 1).otherwise(0)).alias("risco_fogo_nulo"),
        F.sum(F.when(F.col("frp_mw").isNull(), 1).otherwise(0)).alias("frp_nulo"),
        F.sum(F.when(F.col("satelite_referencia"), 1).otherwise(0)).alias("focos_satelite_referencia"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Checagens de sanidade (devem passar; se falharem, a regra está errada ou o dado mudou)

# COMMAND ----------

checks = {
    "risco_fogo dentro de [0,1] ou nulo": s.filter((F.col("risco_fogo") < 0) | (F.col("risco_fogo") > 1)).count() == 0,
    "nenhum -999 sobrevivente": s.filter(
        (F.col("numero_dias_sem_chuva") == -999) | (F.col("precipitacao_mm") == -999) | (F.col("risco_fogo") == -999)
    ).count() == 0,
    "ids únicos": s.select("id").distinct().count() == n_silver,
    "todas as UFs reconhecidas": s.filter(F.col("uf").isNull() & F.col("cod_ibge_uf").isNotNull()).count() == 0,
    "satélite de referência presente": s.filter("satelite_referencia").count() > 0,
    "pelo menos 95% das coordenadas válidas": s.filter("coord_valida").count() / n_silver >= 0.95,
}
for nome, ok in checks.items():
    print(("✓" if ok else "✗"), nome)
assert all(checks.values()), "Alguma checagem falhou — veja acima."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Olhar a Silver antes da EDA

# COMMAND ----------

display(s.orderBy("data_hora_utc").limit(10))

# COMMAND ----------

print("Focos por mês — TODOS os satélites vs. só o de referência (observe como a razão muda ao longo do ano):")
display(s.groupBy("ano", "mes").agg(F.count("*").alias("todos_satelites"), F.sum(F.when(F.col("satelite_referencia"), 1).otherwise(0)).alias("so_aqua_mt")).withColumn("razao", F.round(F.expr("try_divide(todos_satelites, so_aqua_mt)"), 1)).orderBy("ano", "mes"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Dicionário de dados (tabela consultável)

# COMMAND ----------

display(spark.table(TABELA_DICIONARIO))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint da Silver
# MAGIC - [ ] Reconciliação explica a diferença de linhas Bronze → Silver?
# MAGIC - [ ] Todas as checagens passaram?
# MAGIC - [ ] A turma entendeu por que `satelite_referencia` existe e por que FRP não se compara entre sensores?
# MAGIC
# MAGIC Próximo: `03_eda_orientada_hipoteses_exemplo`.