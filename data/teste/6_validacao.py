"""
TESTE DE VALIDAÇÃO: todas as frases (Jeito 1) x método de Lucy, até 500 frases por palavra (Jeito 2).
Só BERT, 2012 e 2018, palavras de gênero, ocupações, inteligência e aparência.

O que é comparado (decidido antes de ver os dados):
  1. cosseno entre o vetor da palavra no Jeito 1 e no Jeito 2 (depois de z-score e L2);
  2. correlação de Pearson entre os vieses do Jeito 1 e do Jeito 2.
     CRITÉRIO: r >= 0,95 no grupo B (critério nosso, declarado como tal);
  3. viés médio de cada grupo (ocupações, inteligência, aparência) nos dois jeitos;
  4. PP1 nos dois jeitos: inclinação e r da regressão viés ~ logit da proporção de mulheres
     no Censo (Garg: occupation_func_female_logitprop) e correlação com o MTurk.

Grupos de palavras:
  A = todas as palavras que existem nos dois jeitos (só informativo);
  B = só as palavras com MAIS de 500 frases, onde o sorteio de fato aconteceu (é onde vale o critério).
As palavras de gênero entram nos itens 1 (cosseno) e na formação dos centroides; o viés (itens 2 a 4)
é calculado para as palavras de ocupações, inteligência e aparência.

O mesmo z-score (zscore.npz do BERT) é usado nos dois jeitos.

Rode a partir da pasta TCC:  python 6_validacao.py
Resultados em resultados/validacao/ (tabelas .csv e resumo.txt).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("metrica", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                       "5_metrica.py"))
metrica = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrica)

PASTAS_VETORES = [os.path.join("data", "vetores_teste"), "vetores_teste"]
MODELO = "bert-base-cased"
ANOS = ["2012", "2018"]
TETO = 500
CRITERIO_R = 0.95
ANOS_CENSO = {"2012": 2012}       # 2018: o Censo do Garg vai só até 2015 (decisão pendente)
SAIDA = os.path.join("resultados", "validacao")


def achar_pasta():
    for p in PASTAS_VETORES:
        if os.path.isdir(os.path.join(p, MODELO)):
            return os.path.join(p, MODELO)
    raise FileNotFoundError(f"Não achei {MODELO} em {PASTAS_VETORES}")


def censo_logit(ano):
    """Logit da proporção de mulheres por ocupação (Garg, utilities.occupation_func_female_logitprop)."""
    d = pd.read_csv(os.path.join(metrica.pasta_listas(), "occupation_percentages_gender_occ1950.csv"))
    d = d[d["Census year"] == ano]
    p = d["Female"].astype(float)
    d = d[(p >= 1e-5) & (p <= 1 - 1e-5)]
    return dict(zip(d["Occupation"].str.strip().str.lower(), np.log(d["Female"] / (1 - d["Female"]))))


def mturk():
    d = pd.read_csv(os.path.join(metrica.pasta_listas(), "mturk_stereotypes.csv"))
    return dict(zip(d["occupation"].str.strip().str.lower(), d["stereotype_score"].astype(float) - 2))  # Garg centra em 2


def regressao(x, y):
    """Inclinação e r de y ~ x (mínimos quadrados, como o OLS do Garg)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    incl = np.polyfit(x, y, 1)[0]
    r = np.corrcoef(x, y)[0, 1]
    return incl, r


def main():
    pasta = achar_pasta()
    os.makedirs(SAIDA, exist_ok=True)
    media, desvio = metrica.carregar_zscore(os.path.join(pasta, "zscore.npz"))
    masc, fem = metrica.ler_lista("male_pairs"), metrica.ler_lista("female_pairs")
    listas = {"ocupacoes": metrica.ler_lista("occupations1950"),
              "inteligencia": metrica.ler_lista("adjectives_intelligencegeneral"),
              "aparencia": metrica.ler_lista("adjectives_appearance")}
    neutras = sorted(set().union(*listas.values()))
    mt = mturk()
    linhas_resumo, tabelas = [], []

    def escrever(txt=""):
        print(txt)
        linhas_resumo.append(txt)

    escrever(f"TESTE DE VALIDAÇÃO — {MODELO} — todas as frases (J1) x até {TETO} frases (J2)")
    escrever(f"Vetores: {pasta} | critério: r >= {CRITERIO_R} entre os vieses, grupo B")

    for ano in ANOS:
        v1_bruto, n1 = metrica.carregar_vetores(os.path.join(pasta, f"completo_{ano}.npz"))
        v2_bruto, n2 = metrica.carregar_vetores(os.path.join(pasta, f"amostra_{ano}.npz"))
        comuns = sorted(set(v1_bruto) & set(v2_bruto))   # J2 já tem o mínimo de 50 frases
        v1 = metrica.pos_processar({p: v1_bruto[p] for p in comuns}, media, desvio)
        v2 = metrica.pos_processar({p: v2_bruto[p] for p in comuns}, media, desvio)

        cm1, cf1, m_ok, f_ok = metrica.centroides(v1, masc, fem)
        cm2, cf2, _, _ = metrica.centroides(v2, masc, fem)
        b1 = metrica.vieses(v1, neutras, cm1, cf1)
        b2 = metrica.vieses(v2, neutras, cm2, cf2)

        tab = pd.DataFrame({"palavra": comuns})
        tab["ano"] = ano
        tab["grupo_lista"] = tab["palavra"].map(
            lambda p: "masculino" if p in masc else "feminino" if p in fem else
            ",".join(n for n, l in listas.items() if p in l))
        tab["frases_J1"] = tab["palavra"].map(n1)
        tab["frases_J2"] = tab["palavra"].map(n2)
        tab["grupo_B"] = tab["frases_J1"] > TETO
        tab["cosseno_J1_J2"] = [float(v1[p] @ v2[p]) for p in comuns]   # vetores já com norma 1
        tab["vies_J1"] = tab["palavra"].map(b1)
        tab["vies_J2"] = tab["palavra"].map(b2)
        tabelas.append(tab)

        escrever(f"\n==================== {ano} ====================")
        escrever(f"Palavras nos dois jeitos: {len(comuns)} (gênero: {len(m_ok)} masc. + {len(f_ok)} fem.; "
                 f"com viés: {tab['vies_J1'].notna().sum()}) | grupo B (> {TETO} frases): {tab['grupo_B'].sum()}")
        escrever(f"Centroides: distância entre o centroide M do J1 e do J2 = {np.linalg.norm(cm1 - cm2):.4f}; "
                 f"F = {np.linalg.norm(cf1 - cf2):.4f}")

        for nome_g, filtro in [("A (todas)", np.ones(len(tab), bool)), ("B (> 500 frases)", tab["grupo_B"].values)]:
            t = tab[filtro]
            tv = t[t["vies_J1"].notna()]
            escrever(f"\n-- Grupo {nome_g} --")
            escrever(f"1. Cosseno J1 x J2: média {t['cosseno_J1_J2'].mean():.4f}, mínimo {t['cosseno_J1_J2'].min():.4f} "
                     f"({t.loc[t['cosseno_J1_J2'].idxmin(), 'palavra']}), n = {len(t)}")
            if len(tv) >= 3:
                r = np.corrcoef(tv["vies_J1"], tv["vies_J2"])[0, 1]
                dif = (tv["vies_J1"] - tv["vies_J2"]).abs()
                escrever(f"2. Correlação dos vieses: r = {r:.4f} (n = {len(tv)})"
                         + (f"  -> {'PASSA' if r >= CRITERIO_R else 'NÃO PASSA'} no critério r >= {CRITERIO_R}"
                            if filtro is not None and nome_g.startswith("B") else ""))
                escrever(f"   Diferença absoluta do viés: média {dif.mean():.4f}, máxima {dif.max():.4f} "
                         f"({tv.loc[dif.idxmax(), 'palavra']}); desvio dos vieses no J1 = {tv['vies_J1'].std():.4f}")
            for lista, palavras in listas.items():
                tl = tv[tv["palavra"].isin(palavras)]
                if len(tl):
                    escrever(f"3. Viés médio {lista:<12} J1 = {tl['vies_J1'].mean():+.4f}  J2 = {tl['vies_J2'].mean():+.4f}"
                             f"  (n = {len(tl)})")

        # 4. PP1 (todas as ocupações com viés nos dois jeitos)
        occ = tab[tab["palavra"].isin(listas["ocupacoes"]) & tab["vies_J1"].notna()]
        escrever("\n-- 4. PP1 (ocupações) --")
        if ano in ANOS_CENSO:
            c = censo_logit(ANOS_CENSO[ano])
            oc = occ[occ["palavra"].isin(c)]
            x = oc["palavra"].map(c)
            i1, r1 = regressao(x, oc["vies_J1"]); i2, r2 = regressao(x, oc["vies_J2"])
            escrever(f"Censo {ANOS_CENSO[ano]} (logit da proporção de mulheres), n = {len(oc)}: "
                     f"J1 inclinação {i1:.4f} r {r1:.3f} | J2 inclinação {i2:.4f} r {r2:.3f}")
        else:
            escrever("Censo: sem ano correspondente no arquivo do Garg (vai até 2015) — decisão pendente.")
        om = occ[occ["palavra"].isin(mt)]
        x = om["palavra"].map(mt)
        i1, r1 = regressao(x, om["vies_J1"]); i2, r2 = regressao(x, om["vies_J2"])
        escrever(f"MTurk (estereótipo percebido), n = {len(om)}: "
                 f"J1 inclinação {i1:.4f} r {r1:.3f} | J2 inclinação {i2:.4f} r {r2:.3f}")

    pd.concat(tabelas).to_csv(os.path.join(SAIDA, "palavras.csv"), index=False)
    with open(os.path.join(SAIDA, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_resumo) + "\n")
    print(f"\nSalvo em {SAIDA}/ (resumo.txt e palavras.csv)")


if __name__ == "__main__":
    main()
