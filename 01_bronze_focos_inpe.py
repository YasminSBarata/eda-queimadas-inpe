# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Bronze — Focos de calor do INPE (Programa Queimadas)
# MAGIC
# MAGIC **Objetivo desta camada:** trazer os arquivos mensais do INPE para dentro do lakehouse
# MAGIC **exatamente como vieram** (todas as colunas como texto), guardando a proveniência de cada linha.
# MAGIC
# MAGIC Bronze não limpa, não tipa, não filtra. Se o arquivo veio com lixo, o lixo entra — a Silver resolve.
# MAGIC Isso permite reprocessar a Silver quantas vezes quisermos sem baixar nada de novo.
# MAGIC
# MAGIC **Fonte:** `https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/mensal/Brasil/`
# MAGIC (arquivos `focos_mensal_br_AAAAMM.csv`, um por mês, Brasil inteiro, todos os satélites).
# MAGIC
# MAGIC **Duas formas de colocar os CSVs no Volume:**
# MAGIC
# MAGIC | Caminho | Quando usar |
# MAGIC | --- | --- |
# MAGIC | **A. Download direto pelo notebook** | Só funciona se a sua conta Free Edition tiver *outbound internet* liberado (hoje isso vem após a verificação via LinkedIn). Tente primeiro — se falhar, vá para B. |
# MAGIC | **B. Upload manual** | Baixe os CSVs no seu computador (links abaixo) e use **Catalog → workspace → queimadas → raw → Upload to this volume**. Depois rode este notebook do mesmo jeito. |
# MAGIC
# MAGIC Tamanhos aproximados dos arquivos de 2025 (todos os satélites): jan 9,7 MB · fev 6,3 · mar 7,5 · abr 4,5 · mai 16 · jun 25 · jul 37 · **ago 89 · set 125 · out 125** · nov 54 · dez 23 → **ano inteiro ≈ 520 MB**.
# MAGIC Para uma turma com upload lento, um subconjunto que preserva a sazonalidade: `202501,202504,202507,202508,202509,202511` (≈ 320 MB).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Parâmetros

# COMMAND ----------

dbutils.widgets.text("catalogo", "workspace", "Catálogo")
dbutils.widgets.text("schema", "queimadas", "Schema")
dbutils.widgets.text("volume", "raw", "Volume (arquivos brutos)")
dbutils.widgets.text(
    "meses",
    "202501,202502,202503,202504,202505,202506,202507,202508,202509,202510,202511,202512",
    "Meses AAAAMM (separados por vírgula)",
)
dbutils.widgets.dropdown("tentar_download", "sim", ["sim", "nao"], "Tentar download direto?")

CATALOGO = dbutils.widgets.get("catalogo").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
VOLUME = dbutils.widgets.get("volume").strip()
MESES = [m.strip() for m in dbutils.widgets.get("meses").split(",") if m.strip()]
TENTAR_DOWNLOAD = dbutils.widgets.get("tentar_download") == "sim"

BASE_URL = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/mensal/Brasil/"
VOLUME_PATH = f"/Volumes/{CATALOGO}/{SCHEMA}/{VOLUME}/inpe_focos"
TABELA_BRONZE = f"{CATALOGO}.{SCHEMA}.bronze_focos_inpe"

print("Volume de arquivos :", VOLUME_PATH)
print("Tabela Bronze      :", TABELA_BRONZE)
print("Meses              :", MESES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Estrutura no Unity Catalog (schema + volume)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOGO}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOGO}.{SCHEMA}.{VOLUME}")

import os
os.makedirs(VOLUME_PATH, exist_ok=True)
print("OK —", VOLUME_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Garantir os arquivos no Volume (caminho A: download; caminho B: já enviados)
# MAGIC
# MAGIC A função abaixo tenta baixar cada mês que ainda não está no Volume. Se o download falhar
# MAGIC (sem internet liberada na Free Edition), ela imprime os links para você baixar e subir manualmente.

# COMMAND ----------

import time
import requests

def nome_arquivo(aaaamm: str) -> str:
    return f"focos_mensal_br_{aaaamm}.csv"

def baixar_mes(aaaamm: str, destino_dir: str, timeout: int = 60) -> str:
    """Baixa o CSV de um mês para o Volume, em streaming. Retorna o caminho final."""
    nome = nome_arquivo(aaaamm)
    url = BASE_URL + nome
    destino = os.path.join(destino_dir, nome)
    tmp = destino + ".part"
    inicio = time.time()
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        baixado = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    baixado += len(chunk)
    os.replace(tmp, destino)
    mb = baixado / 1e6
    print(f"  ✓ {nome}: {mb:,.1f} MB em {time.time() - inicio:,.0f}s")
    return destino

presentes, faltantes, falhas = [], [], []
for m in MESES:
    caminho = os.path.join(VOLUME_PATH, nome_arquivo(m))
    if os.path.exists(caminho) and os.path.getsize(caminho) > 0:
        presentes.append(m)
        continue
    if not TENTAR_DOWNLOAD:
        faltantes.append(m)
        continue
    try:
        print(f"Baixando {m} ...")
        baixar_mes(m, VOLUME_PATH)
        presentes.append(m)
    except Exception as e:  # sem internet liberada, timeout, 404 etc.
        print(f"  ✗ {m}: {type(e).__name__}: {str(e)[:120]}")
        falhas.append(m)

print("\nResumo")
print("  já no Volume / baixados:", presentes)
if faltantes or falhas:
    print("  FALTAM (suba manualmente para", VOLUME_PATH, "):")
    for m in faltantes + falhas:
        print("   ", BASE_URL + nome_arquivo(m))
    print("\n  Como subir: Catalog → workspace → queimadas → raw → pasta inpe_focos → 'Upload to this volume'.")
    print("  Depois, rode este notebook de novo (ele pula o que já existe).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Leitura bruta — tudo como texto
# MAGIC
# MAGIC Lemos com `inferSchema=False` de propósito. Na Bronze queremos ver o dado como ele é:
# MAGIC espaços antes do sinal de menos em `lat`, sentinelas `-999`, `frp` vazio em alguns satélites...
# MAGIC Esses defeitos são **matéria-prima da aula**, não problemas a esconder.

# COMMAND ----------

from pyspark.sql import functions as F, types as T

arquivos = [os.path.join(VOLUME_PATH, nome_arquivo(m)) for m in presentes]
assert arquivos, "Nenhum arquivo disponível no Volume. Veja a célula anterior."

# Schema explícito: as 16 colunas do INPE, todas como texto, + coluna para linhas que não parsearem.
COLUNAS_INPE = [
    "id", "lat", "lon", "data_hora_gmt", "satelite", "municipio", "estado", "pais",
    "municipio_id", "estado_id", "pais_id", "numero_dias_sem_chuva", "precipitacao", "risco_fogo", "bioma", "frp",
]
schema_bronze = T.StructType(
    [T.StructField(c, T.StringType(), True) for c in COLUNAS_INPE]
    + [T.StructField("_linha_corrompida", T.StringType(), True)]
)

df_bronze = (
    spark.read.format("csv")
    .schema(schema_bronze)                # Bronze = texto puro, nada inferido
    .option("header", "true")
    .option("encoding", "UTF-8")
    .option("mode", "PERMISSIVE")         # linhas quebradas não derrubam a carga
    .option("columnNameOfCorruptRecord", "_linha_corrompida")
    .load(arquivos)
    .withColumn("_arquivo_origem", F.col("_metadata.file_name"))   # input_file_name() não funciona no Unity Catalog/serverless
    .withColumn("_ano_mes", F.regexp_extract(F.col("_arquivo_origem"), r"(\d{6})", 1))
    .withColumn("_ingestao_ts", F.current_timestamp())
)

print("Colunas:", df_bronze.columns)
display(df_bronze.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Gravar a tabela Delta Bronze
# MAGIC
# MAGIC `overwrite` completo: a Bronze é reconstruível a partir dos arquivos do Volume a qualquer momento.
# MAGIC (Se preferir cargas incrementais por mês, troque por `replaceWhere` em `_ano_mes`.)

# COMMAND ----------

(
    df_bronze.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("_ano_mes")
    .saveAsTable(TABELA_BRONZE)
)

spark.sql(f"""
COMMENT ON TABLE {TABELA_BRONZE} IS
'Focos de calor do INPE/Programa Queimadas (arquivos mensais, Brasil, todos os satélites). Camada Bronze: colunas como texto, sem limpeza. Fonte: {BASE_URL}'
""")

print("Gravado:", TABELA_BRONZE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Conferência da carga (reconciliação arquivo → tabela)

# COMMAND ----------

contagem = (
    spark.table(TABELA_BRONZE)
    .groupBy("_ano_mes", "_arquivo_origem")
    .agg(F.count("*").alias("linhas"),
         F.sum(F.when(F.col("_linha_corrompida").isNotNull(), 1).otherwise(0)).alias("linhas_corrompidas"))
    .orderBy("_ano_mes")
)
display(contagem)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Primeiro olhar nos defeitos (o que a Silver vai precisar tratar)
# MAGIC
# MAGIC Rode e observe. **Não corrija nada aqui** — anote. Perguntas para a turma:
# MAGIC 1. Por que `lat` tem espaço antes do número? Isso quebra o `CAST`?
# MAGIC 2. O que significa `-999` em `numero_dias_sem_chuva` e em `risco_fogo`? É um valor ou uma ausência?
# MAGIC 3. Por que `frp` vem vazio para os satélites METOP?
# MAGIC 4. Quantos satélites existem? Faz sentido somar focos de todos eles?

# COMMAND ----------

b = spark.table(TABELA_BRONZE)

print("Exemplos de lat com espaço à esquerda:")
display(b.filter(F.col("lat").startswith(" ")).select("lat", "lon", "satelite").limit(5))

print("Sentinela -999:")
display(
    b.select(
        F.sum(F.when(F.col("numero_dias_sem_chuva") == "-999", 1).otherwise(0)).alias("dias_sem_chuva_eh_-999"),
        F.sum(F.when(F.col("risco_fogo").isin("-999", "-999.0"), 1).otherwise(0)).alias("risco_fogo_eh_-999"),
        F.sum(F.when(F.col("precipitacao").isin("-999", "-999.0"), 1).otherwise(0)).alias("precipitacao_eh_-999"),
        F.sum(F.when((F.col("frp").isNull()) | (F.trim(F.col("frp")) == ""), 1).otherwise(0)).alias("frp_vazio"),
        F.count("*").alias("total"),
    )
)

print("Focos por satélite (todos somados — note a diferença de escala entre eles):")
display(b.groupBy("satelite").count().orderBy(F.desc("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint da Bronze
# MAGIC - [ ] Todos os meses desejados aparecem na reconciliação?
# MAGIC - [ ] `linhas_corrompidas` é zero (ou desprezível)?
# MAGIC - [ ] Anotamos os 4 defeitos acima para tratar na Silver?
# MAGIC
# MAGIC Próximo: `02_silver_focos_inpe`.