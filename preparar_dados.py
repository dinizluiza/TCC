"""
Prepara o corpus da Machosfera (Reddit, Ribeiro et al. 2021) para a extração com Transformers.

Diferenças em relação à PoC (que era pensada para word2vec):
  - NÃO põe em minúsculo e NÃO tira pontuação: BERT/RoBERTa/ModernBERT precisam do texto natural.
  - Remove só ruído técnico: entidades HTML, links, marcação de citação, [deleted]/[removed].
  - Usa todos os anos (2008-2018) e o dataset inteiro, sem amostra.
  - Lê o parquet em lotes, para não estourar a memória com 22 milhões de posts.
  - Mantém colunas de metadados (subreddit, comunidade etc.) e salva um parquet por ano.
  - No fim, tira duplicatas exatas dentro de cada ano e imprime quantos posts sobraram por ano.

Antes de rodar: ajuste os nomes de coluna na seção CONFIGURAÇÃO (o script imprime o schema).
"""
import html
import os
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------- CONFIGURAÇÃO
ARQUIVO = os.path.join("data", "reddit.parquet")   # gerado pelo converter_ndjson.py
PASTA_SAIDA = os.path.join("data", "dados_por_ano")
ANO_INICIAL, ANO_FINAL = 2008, 2018

COL_DATA = "date_post"    # data do post
COL_TEXTO = "text_post"   # texto do post/comentário
# Colunas que vale guardar, se existirem no seu parquet (as que não existirem são ignoradas):
COLS_EXTRAS = ["id_post", "subreddit", "thread", "author"]  # campos do reddit.ndjson de Ribeiro
AUTORES_BOT = {"AutoModerator"}  # só é usado se houver coluna "author"
TAMANHO_MINIMO = 10       # caracteres (como no pré-projeto)
TAMANHO_LOTE = 200_000    # linhas por lote; diminua se faltar memória

# Mapa subreddit -> comunidade, do repositório de Ribeiro. O reddit.ndjson também traz subreddits
# que NÃO são da Machosfera (TheBluePill, exredpill = "criticism"; depression, socialanxiety,
# SuicideWatch = "health"; IncelTears, que nem está no mapa). Ribeiro os remove; fazemos igual,
# mantendo só as 5 comunidades abaixo.
ARQUIVO_CATEGORIAS = os.path.join("manosphere_analysis", "data", "subreddit_categories.txt")
COMUNIDADES = {"incels", "mgtow", "mra", "pua", "trp"}

# ----------------------------------------------------------------------------- LIMPEZA
# Citações ("> texto de outra pessoa"): True remove a linha citada inteira, para cada texto conter
# só as palavras do próprio autor (o post citado já está no dataset por conta própria).
# False mantém o texto citado e tira só o sinal ">".
REMOVER_CITACOES = True

LINHA_CITADA = re.compile(r"(?m)^[ \t]*(&gt;|>).*$")      # linha inteira que começa com ">"
MARCA_CITACAO = re.compile(r"(?m)^[ \t]*(&gt;|>)+[ \t]?")  # só o sinal ">" no início da linha
LINK_MARKDOWN = re.compile(r"\[([^\]]*)\]\([^)]*\)")      # [texto](url) -> texto
URL = re.compile(r"https?://\S+|www\.\S+")
ESCAPE_MARKDOWN = re.compile(r"\\([\\`*_{}\[\]()#+\-.!/~>|^])")   # "\/r\/incels" -> "/r/incels"
SOBRESCRITO = re.compile(r"\^\(([^)]*)\)")                 # "^(texto)" -> "texto"
ENFASE = re.compile(r"(?<!\w)(\*{1,3}|~~)(?=\S)(.+?)(?<=\S)\1(?!\w)")     # "*texto*", "**texto**", "~~texto~~" -> "texto"
ESPACOS = re.compile(r"\s+")
REMOVIDOS = {"[deleted]", "[removed]"}
# Mensagens automáticas de bots (ex.: TotesMessenger: "I'm a bot, *bleep*, *bloop*")
TEXTO_BOT = r"\bI(?:'|’| a)?m a bot\b|this action was performed automatically"


def limpar(texto):
    """Tira só ruído técnico; mantém maiúsculas, pontuação e estrutura da frase."""
    if not isinstance(texto, str) or texto.strip() in REMOVIDOS:
        return ""
    texto = (LINHA_CITADA if REMOVER_CITACOES else MARCA_CITACAO).sub("", texto)
    texto = html.unescape(html.unescape(texto))   # 2x: o Reddit às vezes codifica duas vezes (&amp;nbsp;)
    texto = LINK_MARKDOWN.sub(r"\1", texto)
    texto = URL.sub(" ", texto)
    texto = ESCAPE_MARKDOWN.sub(r"\1", texto)
    texto = SOBRESCRITO.sub(r"\1", texto).replace("^", " ")
    texto = ENFASE.sub(r"\2", texto)
    texto = texto.replace("\u200b", " ")           # espaço invisível comum no Reddit
    return ESPACOS.sub(" ", texto).strip()


def para_ano(coluna):
    """Converte a coluna de data para ano, seja ela timestamp Unix (s ou ms), texto ou datetime."""
    if pd.api.types.is_numeric_dtype(coluna):
        unidade = "ms" if coluna.dropna().median() > 1e11 else "s"
        datas = pd.to_datetime(coluna, unit=unidade, errors="coerce")
    else:
        datas = pd.to_datetime(coluna, errors="coerce")
    return datas.dt.year


# ----------------------------------------------------------------------------- 1ª PASSADA
def separar_por_ano():
    arquivo = pq.ParquetFile(ARQUIVO)
    print("Schema do parquet:\n", arquivo.schema_arrow, "\n")

    nomes = set(arquivo.schema_arrow.names)
    faltando = {COL_DATA, COL_TEXTO} - nomes
    if faltando:
        raise SystemExit(f"Colunas não encontradas: {faltando}. Ajuste COL_DATA/COL_TEXTO.")
    if "subreddit" not in nomes:
        raise SystemExit("Coluna 'subreddit' não encontrada: ela é necessária para filtrar as comunidades.")
    extras = [c for c in COLS_EXTRAS if c in nomes]
    print("Colunas extras mantidas:", extras)

    categorias = pd.read_csv(ARQUIVO_CATEGORIAS)
    mapa = dict(zip(categorias["subreddits"].str.strip().str.lower(), categorias["category"].str.strip()))

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    escritores = {}
    contagem = {"lidos": 0, "sem_data": 0, "fora_do_periodo": 0, "fora_da_machosfera": 0,
                "vazios_ou_curtos": 0, "bots": 0}
    removidos_por_subreddit = {}

    for lote in arquivo.iter_batches(batch_size=TAMANHO_LOTE, columns=[COL_DATA, COL_TEXTO] + extras):
        df = lote.to_pandas()
        contagem["lidos"] += len(df)

        df["year"] = para_ano(df[COL_DATA])
        contagem["sem_data"] += int(df["year"].isna().sum())
        no_periodo = df["year"].between(ANO_INICIAL, ANO_FINAL)
        contagem["fora_do_periodo"] += int((~no_periodo & df["year"].notna()).sum())
        df = df[no_periodo].copy()
        df["year"] = df["year"].astype("int16")

        df["categoria"] = df["subreddit"].str.lower().map(mapa)
        fora = ~df["categoria"].isin(COMUNIDADES)
        contagem["fora_da_machosfera"] += int(fora.sum())
        for sub, n in df.loc[fora, "subreddit"].value_counts().items():
            removidos_por_subreddit[sub] = removidos_por_subreddit.get(sub, 0) + int(n)
        df = df[~fora]

        bot = df[COL_TEXTO].str.contains(TEXTO_BOT, case=False, regex=True, na=False)
        if "author" in df.columns:
            bot |= df["author"].isin(AUTORES_BOT)
        contagem["bots"] += int(bot.sum())
        df = df[~bot]

        df["texto"] = df[COL_TEXTO].map(limpar)
        curto = df["texto"].str.len() < TAMANHO_MINIMO
        contagem["vazios_ou_curtos"] += int(curto.sum())
        df = df[~curto].drop(columns=[COL_TEXTO])

        # Todas as colunas de texto como string, para o schema ser igual em todos os lotes
        for c in df.columns:
            if c != "year" and c != COL_DATA:
                df[c] = df[c].astype("string")

        for ano, grupo in df.groupby("year"):
            tabela = pa.Table.from_pandas(grupo, preserve_index=False)
            if ano not in escritores:
                caminho = os.path.join(PASTA_SAIDA, f"{ano}.parquet")
                escritores[ano] = pq.ParquetWriter(caminho, tabela.schema)
            escritores[ano].write_table(tabela.cast(escritores[ano].schema))

        print(f"  {contagem['lidos']:,} linhas lidas...")

    for w in escritores.values():
        w.close()
    print("\nResumo da 1ª passada:", contagem)
    print("Subreddits removidos (fora das 5 comunidades):", removidos_por_subreddit)


# ----------------------------------------------------------------------------- 2ª PASSADA
def tirar_duplicatas_e_resumir():
    """Remove textos idênticos dentro de cada ano (copypasta, bots) e imprime o tamanho de cada ano."""
    resumo = []
    for ano in range(ANO_INICIAL, ANO_FINAL + 1):
        caminho = os.path.join(PASTA_SAIDA, f"{ano}.parquet")
        if not os.path.exists(caminho):
            resumo.append({"ano": ano, "posts": 0, "duplicatas": 0, "palavras": 0})
            continue
        df = pd.read_parquet(caminho)
        antes = len(df)
        df = df.drop_duplicates(subset="texto")
        df.to_parquet(caminho, index=False)
        linha = {
            "ano": ano,
            "posts": len(df),
            "duplicatas": antes - len(df),
            "palavras": int(df["texto"].str.split().str.len().sum()),
        }
        linha.update(df["categoria"].value_counts().to_dict())   # posts por comunidade
        resumo.append(linha)
    resumo = pd.DataFrame(resumo).fillna(0)
    print("\nPosts por ano (depois da limpeza):\n", resumo.to_string(index=False))
    resumo.to_csv(os.path.join(PASTA_SAIDA, "resumo_por_ano.csv"), index=False)


if __name__ == "__main__":
    separar_por_ano()
    tirar_duplicatas_e_resumir()
    print("\nPronto. Um parquet por ano em", PASTA_SAIDA)
