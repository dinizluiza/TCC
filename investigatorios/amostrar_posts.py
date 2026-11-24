"""
Confere a limpeza do preparar_dados.py: sorteia posts de um ano e mostra, um embaixo do outro,
o texto ORIGINAL (data/reddit.parquet) e o LIMPO (data/dados_por_ano/<ano>.parquet).

Duas partes:
  1. Posts sorteados ao acaso, divididos igualmente entre as comunidades do ano.
  2. Posts que tinham algo para limpar (link, &amp;, &gt;, citação), para ver a limpeza agindo.

Rode a partir da pasta TCC:
    python investigatorios/amostrar_posts.py              # 2018, 30 + 10 posts
    python investigatorios/amostrar_posts.py 2013 30      # outro ano / quantidade
Salva em investigatorios/amostra_<ano>.txt
"""
import os
import sys

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

ANO = int(sys.argv[1]) if len(sys.argv) > 1 else 2018
N_ALEATORIOS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
N_COM_SUJEIRA = 10
SEMENTE = 42
MAX_CARACTERES = 700   # corta textos longos na exibição

ANUAL = os.path.join("data", "dados_por_ano", f"{ANO}.parquet")
ORIGINAL = os.path.join("data", "reddit.parquet")
SAIDA = os.path.join("investigatorios", f"amostra_{ANO}.txt")
SUJEIRA = r"https?://|www\.|&amp;|&gt;|&lt;|\]\(|(^|\n)\s*>"

inicio_ano = int(pd.Timestamp(f"{ANO}-01-01", tz="UTC").timestamp())
fim_ano = int(pd.Timestamp(f"{ANO + 1}-01-01", tz="UTC").timestamp())
original = ds.dataset(ORIGINAL)
no_ano = (ds.field("date_post") >= inicio_ano) & (ds.field("date_post") < fim_ano)


def cortar(texto):
    texto = str(texto).replace("\n", " ⏎ ")   # mostra as quebras de linha do original como ⏎
    return texto if len(texto) <= MAX_CARACTERES else texto[:MAX_CARACTERES] + " […]"


def bloco(i, sub, cat, orig, limpo):
    orig = orig if isinstance(orig, str) else "(original não encontrado)"
    limpo = limpo if isinstance(limpo, str) else None
    mudou = "sim" if (limpo is not None and orig.strip() != limpo) else "não"
    linhas = [f"--- #{i} | r/{sub} | {cat} | limpeza alterou o texto? {mudou}",
              f"ORIGINAL: {cortar(orig)}",
              f"LIMPO:    {cortar(limpo)}" if limpo is not None else
              "LIMPO:    (removido: curto, duplicado, bot ou fora das 5 comunidades)",
              ""]
    return "\n".join(linhas)


saida = [f"Amostra do ano {ANO}\n"]

# ----------------------------------------------------------------------------- 1. ALEATÓRIOS
limpos = pd.read_parquet(ANUAL, columns=["id_post", "subreddit", "categoria", "texto"])
comunidades = sorted(limpos["categoria"].dropna().unique())
por_comunidade = max(1, N_ALEATORIOS // len(comunidades))
partes = []
for cat in comunidades:
    grupo = limpos[limpos["categoria"] == cat]
    partes.append(grupo.sample(min(len(grupo), por_comunidade), random_state=SEMENTE))
amostra = pd.concat(partes)

ids = amostra["id_post"].tolist()
originais = (original.to_table(columns=["id_post", "subreddit", "text_post"],
                               filter=no_ano & ds.field("id_post").isin(ids))
             .to_pandas().drop_duplicates(["id_post", "subreddit"]))
amostra = amostra.merge(originais, on=["id_post", "subreddit"], how="left")

saida.append(f"=== 1. {len(amostra)} POSTS ALEATÓRIOS ({por_comunidade} por comunidade: {', '.join(comunidades)}) ===\n")
for i, r in enumerate(amostra.itertuples(), 1):
    saida.append(bloco(i, r.subreddit, r.categoria, r.text_post, r.texto))

# ----------------------------------------------------------------------------- 2. COM SUJEIRA
sujos = (original.to_table(columns=["id_post", "subreddit", "text_post"],
                           filter=no_ano & pc.match_substring_regex(ds.field("text_post"), SUJEIRA))
         .to_pandas())
sujos = sujos.sample(min(len(sujos), N_COM_SUJEIRA), random_state=SEMENTE)
sujos = sujos.merge(limpos, on=["id_post", "subreddit"], how="left")

saida.append(f"\n=== 2. {len(sujos)} POSTS QUE TINHAM LINK, ENTIDADE HTML OU CITAÇÃO ===\n")
for i, r in enumerate(sujos.itertuples(), 1):
    limpo = r.texto if isinstance(r.texto, str) else None
    cat = r.categoria if isinstance(r.categoria, str) else "-"
    saida.append(bloco(i, r.subreddit, cat, r.text_post, limpo))

texto = "\n".join(saida)
sys.stdout.reconfigure(errors="replace")
print(texto)
with open(SAIDA, "w", encoding="utf-8") as f:
    f.write(texto)
print(f"\nSalvo em {SAIDA}")
