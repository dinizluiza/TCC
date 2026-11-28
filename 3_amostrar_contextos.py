"""
Monta as FRASES que vão para o BERT/RoBERTa/ModernBERT, seguindo o protocolo de Lucy et al. (2022)
(repositório context_semantic_axes, code/prep_embedding_data.py):

  1. cada post é quebrado em frases com nltk.sent_tokenize;
  2. cada frase é quebrada em palavras com o BasicTokenizer do BERT, em minúsculas;
  3. frases com menos de 5 ou mais de 150 palavras são descartadas (frases "válidas" = 5 a 150);
  4. para cada palavra-alvo e período, considera-se o CONJUNTO de frases válidas que a contêm
     (a frase conta uma vez, mesmo com a palavra repetida nela);
  5. se forem até 500 frases, usam-se todas; se forem mais, sorteiam-se 500 (amostragem uniforme,
     aqui feita por reservatório, que dá o mesmo resultado que sortear do conjunto inteiro).

Diferença em relação a Lucy: o sorteio é por palavra × PERÍODO (2008-10 agrupado, depois anual),
sem separar por comunidade.

Gera em data/contextos/:
  - contextos_amostra.parquet      frases sorteadas (Jeito 2). Uma linha por frase, com as palavras-alvo
                                   para as quais ela foi sorteada.
  - contextos_completo_2012.parquet  TODAS as frases válidas de 2012 com palavras de gênero, ocupações,
                                     inteligência ou aparência (Jeito 1, teste de validação)
  - contextos_completo_2018.parquet  idem para 2018
  - frases_zscore.parquet          frases quaisquer, sorteadas de posts quaisquer, para calcular as
                                   médias e desvios do z-score (como Lucy fez com a Wikipedia)
  - contagem_frases.csv            nº de frases válidas por palavra e período (para o mínimo de 50)

Precisa de: python -m pip install nltk transformers
Rode a partir da pasta TCC:  python 3_amostrar_contextos.py
"""
import os
import random
import sys
import time
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "investigatorios")           # onde está o palavras_alvo.py
from palavras_alvo import carregar_listas, tokenizar  # noqa: E402

# ----------------------------------------------------------------------------- CONFIGURAÇÃO
PASTA_ANOS = os.path.join("data", "dados_por_ano")
PASTA_SAIDA = os.path.join("data", "contextos")
PERIODOS = {"2008-10": [2008, 2009, 2010], **{str(a): [a] for a in range(2011, 2019)}}
TETO = 500                    # Lucy et al. (2022)
MIN_PALAVRAS, MAX_PALAVRAS = 5, 150   # Lucy et al. (2022)
MINIMO = 50                   # Garg et al. (2018): word1lims = [50, ...]
ANOS_COMPLETOS = {2012, 2018} # períodos do teste de validação (Jeito 1)
LISTAS_TESTE = ["masculino", "feminino", "ocupacoes", "inteligencia", "aparencia"]  # sem traços e Williams & Best
POSTS_ZSCORE = 12_000         # posts sorteados por período para as estatísticas do z-score
SEMENTE = 42
PROCESSOS = 3                 # quantos períodos processar ao mesmo tempo (diminua se faltar memória)
TAMANHO_LOTE = 50_000


def preparar_nltk():
    import nltk
    for recurso in ["punkt_tab", "punkt"]:
        try:
            nltk.data.find(f"tokenizers/{recurso}")
        except LookupError:
            nltk.download(recurso, quiet=True)


def frases_validas(texto, sent_tokenize, basic):
    """Frases de 5 a 150 palavras do texto, cada uma com o conjunto de suas palavras (minúsculas)."""
    for frase in sent_tokenize(texto):
        palavras = basic.tokenize(frase)
        if MIN_PALAVRAS <= len(palavras) <= MAX_PALAVRAS:
            yield frase, palavras


def processar_periodo(periodo):
    """Lê os anos do período e devolve as frases sorteadas, as contagens e a amostra do z-score."""
    from nltk import sent_tokenize
    from transformers import BasicTokenizer

    inicio = time.time()
    basic = BasicTokenizer(do_lower_case=True)
    alvo = set().union(*carregar_listas().values())
    rng = random.Random(f"{SEMENTE}-{periodo}")

    alvo_teste = set().union(*(carregar_listas()[n] for n in LISTAS_TESTE))
    reservatorio = {}          # palavra -> [(frase_id, frase), ...] com até TETO itens
    vistas = {}                # palavra -> nº de frases válidas vistas até agora
    posts_z, n_posts = [], 0   # reservatório de posts quaisquer para o z-score

    escritor_completo, buffer_completo = None, []
    caminho_completo = os.path.join(PASTA_SAIDA, f"contextos_completo_{periodo}.parquet")
    completo = len(PERIODOS[periodo]) == 1 and PERIODOS[periodo][0] in ANOS_COMPLETOS

    def esvaziar_buffer():
        nonlocal escritor_completo, buffer_completo
        if not buffer_completo:
            return
        tabela = pa.Table.from_pandas(pd.DataFrame(buffer_completo), preserve_index=False)
        if escritor_completo is None:
            escritor_completo = pq.ParquetWriter(caminho_completo, tabela.schema, compression="zstd")
        escritor_completo.write_table(tabela)
        buffer_completo = []

    for ano in PERIODOS[periodo]:
        arquivo = pq.ParquetFile(os.path.join(PASTA_ANOS, f"{ano}.parquet"))
        n_linha = 0
        for lote in arquivo.iter_batches(batch_size=TAMANHO_LOTE, columns=["texto"]):
            for texto in lote.column("texto").to_pylist():
                n_linha += 1
                # amostra de posts quaisquer para o z-score (reservatório)
                n_posts += 1
                if len(posts_z) < POSTS_ZSCORE:
                    posts_z.append(texto)
                else:
                    j = rng.randrange(n_posts)
                    if j < POSTS_ZSCORE:
                        posts_z[j] = texto
                # filtro rápido: só quebra em frases os posts que têm alguma palavra-alvo
                if not alvo.intersection(tokenizar(texto)):
                    continue
                for k, (frase, palavras) in enumerate(frases_validas(texto, sent_tokenize, basic)):
                    achadas = alvo.intersection(palavras)
                    if not achadas:
                        continue
                    frase_id = f"{ano}-{n_linha}-{k}"
                    for p in achadas:   # cada frase conta 1 vez por palavra (conjunto, como Lucy)
                        n = vistas.get(p, 0) + 1
                        vistas[p] = n
                        lista = reservatorio.setdefault(p, [])
                        if len(lista) < TETO:
                            lista.append((frase_id, frase))
                        else:
                            j = rng.randrange(n)
                            if j < TETO:
                                lista[j] = (frase_id, frase)
                    achadas_teste = achadas & alvo_teste
                    if completo and achadas_teste:
                        buffer_completo.append({"periodo": periodo, "frase_id": frase_id, "frase": frase,
                                                "palavras": " ".join(sorted(achadas_teste))})
                        if len(buffer_completo) >= 200_000:
                            esvaziar_buffer()
        print(f"  {ano}: {n_linha:,} posts lidos ({time.time() - inicio:.0f} s)", flush=True)

    if completo:
        esvaziar_buffer()
        escritor_completo.close()

    # frases para o z-score: uma frase válida sorteada de cada post sorteado
    frases_z = []
    for i, texto in enumerate(posts_z):
        validas = [f for f, _ in frases_validas(texto, sent_tokenize, basic)]
        if validas:
            frases_z.append({"periodo": periodo, "frase_id": f"z-{periodo}-{i}", "frase": rng.choice(validas)})

    contagem = [{"periodo": periodo, "palavra": p, "frases": n} for p, n in vistas.items()]
    print(f"{periodo}: pronto em {time.time() - inicio:.0f} s", flush=True)
    return periodo, reservatorio, contagem, frases_z


def main():
    preparar_nltk()
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    print(f"Processando {len(PERIODOS)} períodos, {PROCESSOS} de cada vez...")
    with Pool(PROCESSOS) as pool:
        resultados = pool.map(processar_periodo, list(PERIODOS), chunksize=1)

    linhas_amostra, contagens, frases_z = {}, [], []
    for periodo, reservatorio, contagem, fz in resultados:
        contagens += contagem
        frases_z += fz
        validas = {c["palavra"] for c in contagem if c["frases"] >= MINIMO}
        for palavra, lista in reservatorio.items():
            if palavra not in validas:
                continue                       # abaixo do mínimo: não vai para a extração
            for frase_id, frase in lista:      # junta palavras sorteadas na mesma frase
                linha = linhas_amostra.setdefault(frase_id, {"periodo": periodo, "frase_id": frase_id,
                                                             "frase": frase, "palavras": set()})
                linha["palavras"].add(palavra)

    amostra = pd.DataFrame([{**l, "palavras": " ".join(sorted(l["palavras"]))} for l in linhas_amostra.values()])
    amostra.to_parquet(os.path.join(PASTA_SAIDA, "contextos_amostra.parquet"), index=False, compression="zstd")
    pd.DataFrame(frases_z).to_parquet(os.path.join(PASTA_SAIDA, "frases_zscore.parquet"), index=False)
    contagem = pd.DataFrame(contagens).sort_values(["periodo", "palavra"])
    contagem.to_csv(os.path.join(PASTA_SAIDA, "contagem_frases.csv"), index=False)

    pares = amostra["palavras"].str.split().str.len().sum()
    print("\n================ RESUMO ================")
    print(f"Frases únicas na amostra: {len(amostra):,} (pares palavra-frase: {pares:,})")
    print(f"Frases para o z-score: {len(frases_z):,}")
    print("Palavras com pelo menos", MINIMO, "frases válidas, por período:")
    print(contagem[contagem["frases"] >= MINIMO].groupby("periodo")["palavra"].nunique().to_string())
    for ano in sorted(ANOS_COMPLETOS):
        caminho = os.path.join(PASTA_SAIDA, f"contextos_completo_{ano}.parquet")
        print(f"Frases no arquivo completo de {ano}: {pq.ParquetFile(caminho).metadata.num_rows:,}")
    print(f"\nArquivos em {PASTA_SAIDA}/ -> suba essa pasta inteira no Kaggle como dataset")


if __name__ == "__main__":
    main()
