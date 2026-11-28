"""
Conta, para cada ano, comunidade e palavra-alvo:
  - ocorrencias: quantas vezes a palavra aparece (é o que o Garg usa no mínimo de 50);
  - posts: em quantos posts ela aparece pelo menos uma vez.
Também guarda o total de posts e de palavras de cada ano e comunidade.

Lê cada ano UMA vez só, em lotes, e confere todas as palavras-alvo de uma vez.
Salva um arquivo por ano em data/contagem/ (se cair no meio, rodar de novo pula os anos prontos)
e no fim junta tudo em data/contagem/contagem_palavras.csv e data/contagem/totais.csv.

Rode a partir da pasta TCC:
    python contar_palavras.py
"""
import os
import time
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

from palavras_alvo import carregar_listas, tokenizar

PASTA_ANOS = os.path.join("data", "dados_por_ano")
PASTA_SAIDA = os.path.join("data", "contagem")
ANOS = range(2008, 2019)
TAMANHO_LOTE = 100_000

listas = carregar_listas()
alvo = set().union(*listas.values())
print(f"{len(alvo)} palavras-alvo únicas em {len(listas)} listas")
os.makedirs(PASTA_SAIDA, exist_ok=True)

for ano in ANOS:
    arq_contagem = os.path.join(PASTA_SAIDA, f"contagem_{ano}.csv")
    arq_totais = os.path.join(PASTA_SAIDA, f"totais_{ano}.csv")
    if os.path.exists(arq_contagem) and os.path.exists(arq_totais):
        print(f"{ano}: já contado, pulando")
        continue

    inicio = time.time()
    ocorrencias = Counter()   # (categoria, palavra) -> n
    posts = Counter()         # (categoria, palavra) -> n
    total_posts = Counter()   # categoria -> n
    total_palavras = Counter()

    arquivo = pq.ParquetFile(os.path.join(PASTA_ANOS, f"{ano}.parquet"))
    for lote in arquivo.iter_batches(batch_size=TAMANHO_LOTE, columns=["texto", "categoria"]):
        for texto, categoria in zip(lote.column("texto").to_pylist(), lote.column("categoria").to_pylist()):
            palavras = tokenizar(texto)
            total_posts[categoria] += 1
            total_palavras[categoria] += len(palavras)
            achadas = Counter(p for p in palavras if p in alvo)
            for p, n in achadas.items():
                ocorrencias[(categoria, p)] += n
                posts[(categoria, p)] += 1

    linhas = [{"ano": ano, "categoria": c, "palavra": p, "ocorrencias": n, "posts": posts[(c, p)]}
              for (c, p), n in ocorrencias.items()]
    pd.DataFrame(linhas).to_csv(arq_contagem, index=False)
    pd.DataFrame([{"ano": ano, "categoria": c, "posts": total_posts[c], "palavras": total_palavras[c]}
                  for c in total_posts]).to_csv(arq_totais, index=False)
    print(f"{ano}: {sum(total_posts.values()):,} posts, {len(set(p for _, p in ocorrencias))} palavras-alvo"
          f" encontradas, {time.time() - inicio:.0f} s")

contagem = pd.concat(pd.read_csv(os.path.join(PASTA_SAIDA, f"contagem_{a}.csv")) for a in ANOS)
totais = pd.concat(pd.read_csv(os.path.join(PASTA_SAIDA, f"totais_{a}.csv")) for a in ANOS)
contagem.to_csv(os.path.join(PASTA_SAIDA, "contagem_palavras.csv"), index=False)
totais.to_csv(os.path.join(PASTA_SAIDA, "totais.csv"), index=False)
print(f"\nPronto: {PASTA_SAIDA}/contagem_palavras.csv ({len(contagem):,} linhas) e totais.csv")
