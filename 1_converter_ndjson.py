"""
Converte o reddit.ndjson (Ribeiro et al. 2021, Zenodo) em reddit.parquet.

Lê o arquivo em partes de 200 mil linhas, para não estourar a memória,
e guarda só as colunas úteis. Depois rode o preparar_dados.py.

No fim, mostra um "antes e depois": quantos valores nulos cada coluna tinha
antes da conversão de tipos e quantos tem depois. Se a conversão de date_post
tivesse estragado alguma data, a diferença apareceria aqui. O relatório também
fica salvo em data/conversao_relatorio.txt.
"""
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ENTRADA = os.path.join("data", "reddit.ndjson")
SAIDA = os.path.join("data", "reddit.parquet")
RELATORIO = os.path.join("data", "conversao_relatorio.txt")
TAMANHO_PARTE = 200_000   # diminua se faltar memória

COLUNAS_STRING = ["author", "id_post", "subreddit", "text_post", "thread"]
COLUNAS = ["date_post"] + COLUNAS_STRING   # number_post fica de fora (vazia em 92% das linhas)

escritor = None
total = 0
nulos_antes = pd.Series(0, index=COLUNAS)
nulos_depois = pd.Series(0, index=COLUNAS)
datas_em_texto = 0          # quantas datas vieram como texto ("1546300841") em vez de número
exemplos_perdidos = []      # até 10 valores de date_post que viraram nulo na conversão

leitor = pd.read_json(ENTRADA, lines=True, chunksize=TAMANHO_PARTE,
                      dtype=False, convert_dates=False, encoding="utf-8")

for parte in leitor:
    parte = parte.reindex(columns=COLUNAS)

    # ---- ANTES da conversão
    nulos_antes += parte.isna().sum()
    datas_em_texto += int(parte["date_post"].map(lambda v: isinstance(v, str)).sum())
    data_original = parte["date_post"]

    # ---- Conversão
    # date_post vem como número em ~2/3 das linhas e como texto numérico no resto: to_numeric resolve os dois
    parte["date_post"] = pd.to_numeric(parte["date_post"], errors="coerce").astype("Int64")
    for c in COLUNAS_STRING:
        parte[c] = parte[c].astype("string")

    # ---- DEPOIS da conversão
    nulos_depois += parte.isna().sum()
    perdidos = data_original.notna() & parte["date_post"].isna()
    if perdidos.any() and len(exemplos_perdidos) < 10:
        exemplos_perdidos += data_original[perdidos].head(10 - len(exemplos_perdidos)).tolist()

    tabela = pa.Table.from_pandas(parte, preserve_index=False)
    if escritor is None:
        escritor = pq.ParquetWriter(SAIDA, tabela.schema, compression="zstd")
    escritor.write_table(tabela.cast(escritor.schema))

    total += len(parte)
    print(f"  {total:,} linhas convertidas...")

if escritor is not None:
    escritor.close()

# ----------------------------------------------------------------------------- RELATÓRIO
linhas_no_parquet = pq.ParquetFile(SAIDA).metadata.num_rows
comparacao = pd.DataFrame({
    "nulos_antes": nulos_antes,
    "nulos_depois": nulos_depois,
    "criados_na_conversao": nulos_depois - nulos_antes,
})

relatorio = [
    f"Linhas lidas do ndjson:     {total:,}",
    f"Linhas gravadas no parquet: {linhas_no_parquet:,}"
    + ("   OK" if linhas_no_parquet == total else "   ATENÇÃO: número diferente!"),
    f"Datas que vieram como texto e foram convertidas: {datas_em_texto:,}",
    "",
    "Valores nulos por coluna, antes e depois da conversão de tipos:",
    comparacao.to_string(),
    "",
]
perdidas = int(comparacao.loc["date_post", "criados_na_conversao"])
if perdidas == 0:
    relatorio.append("date_post: nenhuma data se perdeu na conversão.")
else:
    relatorio.append(f"date_post: {perdidas:,} datas viraram nulo na conversão. Exemplos dos valores originais:")
    relatorio += [f"  {v!r}" for v in exemplos_perdidos]

texto = "\n".join(relatorio)
print("\n" + texto)
with open(RELATORIO, "w", encoding="utf-8") as f:
    f.write(texto)
print(f"\nPronto: {SAIDA}. Relatório salvo em {RELATORIO}")
