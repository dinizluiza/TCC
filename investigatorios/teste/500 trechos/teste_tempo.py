"""
Teste de tempo: calcula o vetor médio de UMA palavra em UM ano, com UM modelo, e mede cada etapa.

Dois modos, para comparar:
    python investigatorios/teste_tempo.py             COMPLETO: todas as ocorrências, post inteiro
    python investigatorios/teste_tempo.py 500 50      AMOSTRA: 500 trechos sorteados, 50 palavras antes e 50 depois

Palavra, ano e modelo são opcionais:
    python investigatorios/teste_tempo.py 500 50 --palavra nurse --ano 2018 --modelo roberta-base

Cada execução salva o vetor em investigatorios/. Se já existir um vetor da mesma palavra/ano/modelo
feito no outro modo, o script mostra a similaridade entre os dois (perto de 1 = a amostra basta).

Precisa de: pip install torch transformers
(na primeira vez, o modelo é baixado da internet — ~400 MB, e esse download não entra na medição)
"""
import argparse
import glob
import os
import re
import time

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("n_trechos", type=int, nargs="?", help="quantos trechos sortear (vazio = todos)")
parser.add_argument("palavras_em_volta", type=int, nargs="?", help="palavras antes e depois (vazio = post inteiro)")
parser.add_argument("--palavra", default="engineer")
parser.add_argument("--ano", type=int, default=2015)
parser.add_argument("--modelo", default="bert-base-cased")
args = parser.parse_args()

COMPLETO = args.n_trechos is None
JANELA_MAX = 175           # no modo completo, posts longos demais para o modelo (512 tokens) viram 175 palavras de cada lado
TAMANHO_LOTE = 16          # trechos por vez; diminui sozinho se faltar memória na GPU
SEMENTE = 42
N_PALAVRAS_TOTAL = 20 + 20 + 76 + 638 + 28 + 25   # gênero + ocupações + traços + inteligência + aparência
N_ANOS, N_MODELOS = 11, 3

if not COMPLETO and args.palavras_em_volta is None:
    parser.error("no modo amostra, informe os dois números: python teste_tempo.py 500 50")

rotulo = "completo" if COMPLETO else f"n{args.n_trechos}_w{args.palavras_em_volta}"
print(f"Modo: {'COMPLETO (todas as ocorrências, post inteiro)' if COMPLETO else f'AMOSTRA ({args.n_trechos} trechos, {args.palavras_em_volta} palavras de cada lado)'}")
print(f"Palavra '{args.palavra}', ano {args.ano}, modelo {args.modelo}\n")

tempos = {}


class cronometro:
    def __init__(self, nome):
        self.nome = nome

    def __enter__(self):
        self.t = time.perf_counter()
        print(f"> {self.nome}...")

    def __exit__(self, *a):
        tempos[self.nome] = time.perf_counter() - self.t
        print(f"  {tempos[self.nome]:.1f} s")


# ----------------------------------------------------------------------------- 1. LER O ANO
with cronometro("ler o ano"):
    textos = pd.read_parquet(os.path.join("data", "dados_por_ano", f"{args.ano}.parquet"), columns=["texto"])["texto"]
    print(f"  {len(textos):,} posts em {args.ano}")

# ----------------------------------------------------------------------------- 2. ACHAR OCORRÊNCIAS
# Palavra inteira, sem diferenciar maiúsculas: "man" não casa com "manager" nem com "woman"
padrao = re.compile(rf"(?<![\w']){re.escape(args.palavra)}(?![\w'])", re.IGNORECASE)

with cronometro("achar ocorrências"):
    com_palavra = textos[textos.str.contains(padrao, na=False)]
    print(f"  '{args.palavra}' aparece em {len(com_palavra):,} posts")
if com_palavra.empty:
    raise SystemExit("Nenhuma ocorrência. Tente outra palavra ou outro ano.")


# ----------------------------------------------------------------------------- 3. MONTAR OS TRECHOS
def recortar(texto, palavras_em_volta):
    """Janela de N palavras antes e depois da 1ª ocorrência. Devolve (trecho, (início, fim) da palavra no trecho)."""
    m = padrao.search(texto)
    if palavras_em_volta is None:
        return texto, (m.start(), m.end())
    blocos = list(re.finditer(r"\S+", texto))                         # cada "palavra" com sua posição
    i = next(k for k, b in enumerate(blocos) if b.end() > m.start())  # bloco que contém a palavra-alvo
    ini = blocos[max(0, i - palavras_em_volta)].start()
    fim = blocos[min(len(blocos) - 1, i + palavras_em_volta)].end()
    return texto[ini:fim], (m.start() - ini, m.end() - ini)


with cronometro("montar trechos"):
    if COMPLETO:
        escolhidos = com_palavra
    else:
        escolhidos = com_palavra.sample(min(args.n_trechos, len(com_palavra)), random_state=SEMENTE)
    pares = []
    for texto in escolhidos:
        if COMPLETO:
            janela = None if len(texto.split()) <= 2 * JANELA_MAX else JANELA_MAX
        else:
            janela = args.palavras_em_volta
        pares.append(recortar(texto, janela))
    pares.sort(key=lambda p: len(p[0]))     # trechos de tamanho parecido no mesmo lote = menos desperdício
    trechos = [p[0] for p in pares]
    alvos = [p[1] for p in pares]
    tamanhos = [len(t.split()) for t in trechos]
    print(f"  {len(trechos):,} trechos; palavras por trecho: mediana {np.median(tamanhos):.0f}, máximo {max(tamanhos)}")

# ----------------------------------------------------------------------------- 4. CARREGAR O MODELO
import torch
from transformers import AutoModel, AutoTokenizer

dispositivo = "cpu"
if torch.cuda.is_available():
    try:
        torch.zeros(1).cuda() + 1          # confirma que a GPU funciona de verdade com esta versão do torch
        dispositivo = "cuda"
    except Exception as erro:
        print(f"  (GPU encontrada, mas não funcionou com esta versão do torch: {erro}. Usando CPU.)")
print(f"\nDispositivo: {dispositivo} {torch.cuda.get_device_name(0) if dispositivo == 'cuda' else ''}")

with cronometro("carregar o modelo"):
    tokenizador = AutoTokenizer.from_pretrained(args.modelo)
    modelo = AutoModel.from_pretrained(args.modelo).to(dispositivo).eval()
max_tokens = min(512, tokenizador.model_max_length)

# ----------------------------------------------------------------------------- 5. EXTRAIR OS VETORES
vetores, perdidas = [], 0
lote = TAMANHO_LOTE
with cronometro("extrair vetores"), torch.no_grad():
    i = 0
    while i < len(trechos):
        bloco, bloco_alvos = trechos[i:i + lote], alvos[i:i + lote]
        try:
            enc = tokenizador(bloco, return_offsets_mapping=True, truncation=True,
                              max_length=max_tokens, padding=True, return_tensors="pt")
            posicoes = enc.pop("offset_mapping")
            saida = modelo(**{k: v.to(dispositivo) for k, v in enc.items()}).last_hidden_state.cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            lote = max(1, lote // 2)
            print(f"\n  memória da GPU cheia: lote reduzido para {lote}")
            continue
        for b, (a_ini, a_fim) in enumerate(bloco_alvos):
            # tokens cujos caracteres caem dentro da palavra-alvo (1 ou mais pedaços)
            idx = [t for t, (s, e) in enumerate(posicoes[b].tolist()) if e > s and s >= a_ini and e <= a_fim]
            if idx:
                vetores.append(saida[b, idx].mean(0).numpy())
            else:
                perdidas += 1          # palavra caiu fora do limite de tokens
        i += len(bloco)
        print(f"  {i:,}/{len(trechos):,} trechos", end="\r")
    print()

vetor = np.mean(vetores, axis=0)
vetor = vetor / np.linalg.norm(vetor)

# ----------------------------------------------------------------------------- RESULTADO
por_trecho = tempos["extrair vetores"] / len(vetores)
print("\n================ RESULTADO ================")
print(f"Modo {rotulo} | '{args.palavra}' | {args.ano} | {args.modelo} | {dispositivo}")
print(f"Ocorrências usadas: {len(vetores):,} (perdidas por passar do limite de tokens: {perdidas})")
print(f"Vetor: {len(vetor)} números, norma {np.linalg.norm(vetor):.3f}")
print("\nTempos:")
for nome, t in tempos.items():
    print(f"  {nome:<22}{t:>9.1f} s")
print(f"  {'por trecho':<22}{por_trecho * 1000:>9.1f} ms")

trechos_por_palavra = len(vetores)
total = N_PALAVRAS_TOTAL * trechos_por_palavra * N_ANOS * N_MODELOS
horas = total * por_trecho / 3600
print(f"\nProjeção grosseira neste computador, SE toda palavra em todo ano tivesse {trechos_por_palavra:,} trechos como esta:")
print(f"  {N_PALAVRAS_TOTAL} palavras × {trechos_por_palavra:,} × {N_ANOS} anos × {N_MODELOS} modelos = {total:,} trechos"
      f" → ~{horas:.1f} h (~{horas / N_MODELOS:.1f} h por modelo)")
print("  (a busca de ocorrências não entra: no script real ela é feita para todas as palavras numa leitura só de cada ano)")

# ----------------------------------------------------------------------------- COMPARAR COM O OUTRO MODO
os.makedirs("investigatorios", exist_ok=True)
base = f"vetor_{args.palavra}_{args.ano}_{args.modelo.split('/')[-1]}"
np.save(os.path.join("investigatorios", f"{base}_{rotulo}.npy"), vetor)
for arquivo in sorted(glob.glob(os.path.join("investigatorios", f"{base}_*.npy"))):
    outro = os.path.basename(arquivo)[len(base) + 1:-4]
    if outro != rotulo:
        similaridade = float(np.dot(vetor, np.load(arquivo)))   # ambos normalizados: produto = cosseno
        print(f"\nSimilaridade de cosseno entre '{rotulo}' e '{outro}': {similaridade:.4f}")
