"""
Métrica de viés de Garg et al. (2018) aplicada aos vetores extraídos no Kaggle.

Ordem do pós-processamento (decidida no projeto):
  1. z-score por dimensão: (vetor - média) / desvio, com média e desvio do zscore.npz
     (uma palavra sorteada por frase, como Lucy et al. 2022; corrige as dimensões rebeldes,
     Timkey & van Schijndel 2021);
  2. normalização L2 de cada vetor de palavra (Garg et al. 2018, normalize_vectors.py);
  3. Relative Norm Distance (Garg et al. 2018, changes_over_time.py):
       centroide_M = média dos vetores das palavras masculinas
       centroide_F = média dos vetores das palavras femininas
       viés(w) = ||w - centroide_M|| - ||w - centroide_F||
     viés > 0: w está mais perto do polo FEMININO; viés < 0: mais perto do MASCULINO.
     (No código do Garg: differences(dist_male, dist_female), com a distância até o vetor médio
      de cada grupo; índice 4 de single_set_distances_to_single_set.)

Para conferir a métrica com vetores inventados:  python 5_metrica.py
"""
import os

import numpy as np

PASTA_LISTAS_CANDIDATAS = [os.path.join("repos ref", "EmbeddingDynamicStereotypes", "data"),
                           os.path.join("EmbeddingDynamicStereotypes", "data")]


def pasta_listas():
    for c in PASTA_LISTAS_CANDIDATAS:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("Não achei a pasta data do repositório do Garg (EmbeddingDynamicStereotypes).")


def ler_lista(nome):
    with open(os.path.join(pasta_listas(), f"{nome}.txt"), encoding="utf-8") as f:
        return [linha.strip().lower() for linha in f if linha.strip()]


def carregar_vetores(caminho_npz):
    """Devolve (dict palavra -> vetor bruto float64, dict palavra -> nº de frases)."""
    d = np.load(caminho_npz)
    palavras = [str(p) for p in d["palavras"]]
    vetores = {p: d["medias"][i].astype(np.float64) for i, p in enumerate(palavras)}
    n = {p: int(d["n"][i]) for i, p in enumerate(palavras)}
    return vetores, n


def carregar_zscore(caminho_npz):
    """Só a média e o desvio (não carrega os vetores individuais, que são grandes)."""
    d = np.load(caminho_npz)
    return d["media"].astype(np.float64), d["desvio"].astype(np.float64)


def pos_processar(vetores, media=None, desvio=None):
    """Passos 1 e 2: z-score (se media/desvio forem dados) e depois L2."""
    saida = {}
    for p, v in vetores.items():
        if media is not None:
            v = (v - media) / desvio
        saida[p] = v / np.linalg.norm(v)
    return saida


def centroides(vetores, masculinas, femininas):
    """Vetor médio de cada grupo, só com as palavras de gênero que existem nos vetores."""
    m = [p for p in masculinas if p in vetores]
    f = [p for p in femininas if p in vetores]
    if not m or not f:
        raise ValueError("Faltam palavras de gênero para formar os centroides.")
    return np.mean([vetores[p] for p in m], axis=0), np.mean([vetores[p] for p in f], axis=0), m, f


def vieses(vetores, palavras, c_m, c_f):
    """Passo 3: viés de cada palavra (as que existem nos vetores)."""
    return {p: float(np.linalg.norm(vetores[p] - c_m) - np.linalg.norm(vetores[p] - c_f))
            for p in palavras if p in vetores}


def _autoteste():
    print("Autoteste com vetores inventados (2 dimensões, sem z-score):")
    vet = {"he": np.array([1.0, 0.0]), "she": np.array([0.0, 1.0]),
           "nurse": np.array([0.0, 1.0]), "engineer": np.array([1.0, 0.0]), "person": np.array([1.0, 1.0])}
    vet = pos_processar(vet)
    c_m, c_f, _, _ = centroides(vet, ["he"], ["she"])
    b = vieses(vet, ["nurse", "engineer", "person"], c_m, c_f)
    esperado = {"nurse": np.sqrt(2), "engineer": -np.sqrt(2), "person": 0.0}
    for p, v in b.items():
        ok = abs(v - esperado[p]) < 1e-9
        print(f"  {p:<9} viés = {v:+.4f}   esperado {esperado[p]:+.4f}   {'OK' if ok else 'ERRO'}")
        assert ok
    print("  nurse > 0 (perto do feminino), engineer < 0 (perto do masculino), person = 0 (no meio).")

    print("Autoteste do z-score: uma dimensão enorme e igual para todos não pode dominar.")
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 5)); base[:, 0] += 50          # dimensão 0 'rebelde'
    media, desvio = base.mean(0), base.std(0)
    a, b2 = base[0], base[1]
    cos_bruto = a @ b2 / np.linalg.norm(a) / np.linalg.norm(b2)
    za, zb = (a - media) / desvio, (b2 - media) / desvio
    cos_z = za @ zb / np.linalg.norm(za) / np.linalg.norm(zb)
    print(f"  cosseno sem z-score = {cos_bruto:.3f} (inflado pela dimensão rebelde); com z-score = {cos_z:+.3f}")
    assert cos_bruto > 0.99 and abs(cos_z) < 0.99
    print("Tudo OK.")


if __name__ == "__main__":
    _autoteste()
