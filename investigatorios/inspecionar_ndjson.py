"""
Inspeciona o reddit.ndjson ANTES de converter: quantas linhas tem, quais colunas (campos) existem,
de que tipo são, quantos valores vazios, intervalo de datas, posts por ano e por subreddit.

Lê o arquivo linha a linha (não carrega tudo na memória). No arquivo inteiro (4,2 GB)
deve levar alguns minutos. Para um teste rápido, rode com um limite de linhas:

    python inspecionar_ndjson.py                 # arquivo inteiro
    python inspecionar_ndjson.py 100000          # só as primeiras 100 mil linhas

Também salva tudo em inspecao_reddit.txt.
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

ARQUIVO = r"C:\Users\luiza_cv9lc2j\TCC\data\reddit.ndjson"
SAIDA = r"C:\Users\luiza_cv9lc2j\TCC\investigatorios\inspecao_reddit.txt"
N_EXEMPLOS = 3

limite = int(sys.argv[1]) if len(sys.argv) > 1 else None
sys.stdout.reconfigure(errors="replace")   # evita erro no terminal do Windows com emojis nos exemplos

linhas = 0
linhas_vazias = 0
linhas_invalidas = 0
presenca = Counter()                    # em quantas linhas cada campo aparece
vazios = Counter()                      # quantas vezes o campo é null ou ""
tipos = defaultdict(Counter)            # tipos Python encontrados em cada campo
exemplos = []                           # primeiras linhas completas
por_ano = Counter()
por_subreddit = Counter()
datas_invalidas = 0
data_min = data_max = None
tamanho_texto = []                      # amostra de tamanhos do texto (1 a cada 100 linhas)

inicio = time.time()
with open(ARQUIVO, "r", encoding="utf-8") as f:
    for linha in f:
        if limite and linhas >= limite:
            break
        linhas += 1
        if linhas % 1_000_000 == 0:
            print(f"  {linhas:,} linhas lidas ({time.time() - inicio:.0f} s)...")

        linha = linha.strip()
        if not linha:
            linhas_vazias += 1
            continue
        try:
            registro = json.loads(linha)
        except json.JSONDecodeError:
            linhas_invalidas += 1
            continue

        if len(exemplos) < N_EXEMPLOS:
            exemplos.append(registro)

        for campo, valor in registro.items():
            presenca[campo] += 1
            tipos[campo][type(valor).__name__] += 1
            if valor is None or valor == "":
                vazios[campo] += 1

        data = registro.get("date_post")
        try:
            ts = int(float(data))
            if ts > 1e11:          # milissegundos
                ts //= 1000
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            por_ano[dt.year] += 1
            data_min = dt if data_min is None or dt < data_min else data_min
            data_max = dt if data_max is None or dt > data_max else data_max
        except (TypeError, ValueError, OverflowError, OSError):
            datas_invalidas += 1

        por_subreddit[registro.get("subreddit")] += 1

        texto = registro.get("text_post")
        if linhas % 100 == 0 and isinstance(texto, str):
            tamanho_texto.append(len(texto))

# ----------------------------------------------------------------------------- RELATÓRIO
validas = linhas - linhas_vazias - linhas_invalidas
saida = []
p = saida.append

p(f"Arquivo: {ARQUIVO} ({os.path.getsize(ARQUIVO) / 1e9:.2f} GB)")
p(f"Leitura: {'primeiras ' + format(limite, ',') + ' linhas' if limite else 'arquivo inteiro'}"
  f" em {time.time() - inicio:.0f} s")
p(f"Linhas lidas: {linhas:,} | válidas: {validas:,} | vazias: {linhas_vazias:,} | JSON inválido: {linhas_invalidas:,}")

p("\n=== CAMPOS (colunas) ===")
p(f"{'campo':<15}{'presente em':>15}{'% linhas':>10}{'null/vazio':>13}   tipos")
for campo, n in presenca.most_common():
    tipos_txt = ", ".join(f"{t}: {c:,}" for t, c in tipos[campo].most_common())
    p(f"{campo:<15}{n:>15,}{100 * n / max(validas, 1):>9.1f}%{vazios[campo]:>13,}   {tipos_txt}")

p("\n=== DATAS (date_post, UTC) ===")
p(f"Mais antiga: {data_min} | mais recente: {data_max} | inválidas ou ausentes: {datas_invalidas:,}")
p("Posts por ano:")
for ano in sorted(por_ano):
    p(f"  {ano}: {por_ano[ano]:>12,}")

p(f"\n=== SUBREDDITS ({len(por_subreddit)} distintos) ===")
for sub, n in por_subreddit.most_common():
    p(f"  {str(sub):<25}{n:>12,}")

if tamanho_texto:
    tamanho_texto.sort()
    meio = tamanho_texto[len(tamanho_texto) // 2]
    p(f"\n=== TAMANHO DO TEXTO (amostra de {len(tamanho_texto):,} posts) ===")
    p(f"Mediana: {meio} caracteres | mínimo: {tamanho_texto[0]} | máximo: {tamanho_texto[-1]:,}")

p(f"\n=== PRIMEIRAS {len(exemplos)} LINHAS ===")
for ex in exemplos:
    ex = {k: (v[:200] + "…" if isinstance(v, str) and len(v) > 200 else v) for k, v in ex.items()}
    p(json.dumps(ex, ensure_ascii=False, indent=2))

relatorio = "\n".join(saida)
print("\n" + relatorio)
with open(SAIDA, "w", encoding="utf-8") as f:
    f.write(relatorio)
print(f"\nRelatório salvo em {SAIDA}")
