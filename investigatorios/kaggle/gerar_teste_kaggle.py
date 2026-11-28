"""
Prepara os arquivos do teste de tempo no Kaggle.

Gera os MESMOS 500 trechos de "engineer" em 2015 do teste que você rodou no seu computador
(mesmo sorteio, mesma semente, 50 palavras de cada lado) e copia o vetor que a MX330 calculou.
No Kaggle, o notebook mede o tempo e confere se o vetor de lá bate com o daqui.

Rode a partir da pasta TCC:
    python investigatorios/kaggle/gerar_teste_kaggle.py
Os arquivos para subir no Kaggle ficam em investigatorios/kaggle/para_subir/
"""
import glob
import os
import re
import shutil

import pandas as pd

PALAVRA, ANO, N_TRECHOS, PALAVRAS_EM_VOLTA, SEMENTE = "engineer", 2015, 500, 50, 42
SAIDA = os.path.join("investigatorios", "kaggle", "para_subir")
os.makedirs(SAIDA, exist_ok=True)

# Mesma regra e mesmo sorteio do teste_tempo.py, para comparar com o vetor da MX330
padrao = re.compile(rf"(?<![\w']){re.escape(PALAVRA)}(?![\w'])", re.IGNORECASE)
textos = pd.read_parquet(os.path.join("data", "dados_por_ano", f"{ANO}.parquet"), columns=["texto"])["texto"]
com_palavra = textos[textos.str.contains(padrao, na=False)]
amostra = com_palavra.sample(min(N_TRECHOS, len(com_palavra)), random_state=SEMENTE)

linhas = []
for texto in amostra:
    m = padrao.search(texto)
    blocos = list(re.finditer(r"\S+", texto))
    i = next(k for k, b in enumerate(blocos) if b.end() > m.start())
    ini = blocos[max(0, i - PALAVRAS_EM_VOLTA)].start()
    fim = blocos[min(len(blocos) - 1, i + PALAVRAS_EM_VOLTA)].end()
    linhas.append({"trecho": texto[ini:fim], "ini": m.start() - ini, "fim": m.end() - ini})

arquivo = os.path.join(SAIDA, f"trechos_{PALAVRA}_{ANO}.csv")
pd.DataFrame(linhas).to_csv(arquivo, index=False)
print(f"{len(linhas)} trechos salvos em {arquivo}")

referencia = glob.glob(os.path.join("investigatorios", "**", f"vetor_{PALAVRA}_{ANO}_bert-base-cased_n500_w50.npy"),
                       recursive=True)
if referencia:
    shutil.copy(referencia[0], os.path.join(SAIDA, "vetor_referencia_mx330.npy"))
    print(f"Vetor de referência copiado de {referencia[0]}")
else:
    print("Vetor de referência da MX330 não encontrado (o teste roda igual, só sem essa comparação).")
print(f"\nSuba no Kaggle os arquivos da pasta {SAIDA}")
