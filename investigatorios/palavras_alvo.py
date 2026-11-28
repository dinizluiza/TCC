"""
Definições compartilhadas por contar_palavras.py e, depois, pelo sorteio de trechos:
  - quais listas de palavras usamos (do repositório do Garg);
  - o que conta como "uma ocorrência" de uma palavra.

REGRA DE OCORRÊNCIA (a mesma em todos os scripts):
  o texto é posto em minúsculas e quebrado em sequências de letras. Qualquer coisa que não é letra
  (espaço, pontuação, número, apóstrofo, hífen) separa palavras. Assim:
    "Engineer," -> engineer      "women's" -> women + s      "she's" -> she + s
    "manager"   -> NÃO conta como "man"                    "woman" -> NÃO conta como "man"
"""
import os
import re

# o repositório do Garg pode estar em TCC/ ou em TCC/repos ref/ (procura nos dois)
_CANDIDATAS = [os.path.join("EmbeddingDynamicStereotypes", "data"),
               os.path.join("repos ref", "EmbeddingDynamicStereotypes", "data")]
PASTA_LISTAS = next((c for c in _CANDIDATAS if os.path.isdir(c)), _CANDIDATAS[0])

# nome que usamos -> arquivo do Garg
LISTAS = {
    "masculino": "male_pairs",
    "feminino": "female_pairs",
    "ocupacoes": "occupations1950",                     # PP1 (contém as duas listas abaixo)
    "ocupacoes_profissionais": "occupations1950_professional",
    "ocupacoes_mturk": "occupationsMturk",
    "inteligencia": "adjectives_intelligencegeneral",   # PP2
    "aparencia": "adjectives_appearance",               # PP2
    "tracos_personalidade": "personalitytraits_original",  # PP3, opção (a)
    "williams_best": "adjectives_williamsbest",            # PP3, opção (b)
}

_LETRAS = re.compile(r"[^\W\d_]+")   # sequência de letras (inclui acentos)


def carregar_listas():
    """Devolve {nome_da_lista: [palavras]}."""
    listas = {}
    for nome, arquivo in LISTAS.items():
        with open(os.path.join(PASTA_LISTAS, f"{arquivo}.txt"), encoding="utf-8") as f:
            listas[nome] = [linha.strip().lower() for linha in f if linha.strip()]
    return listas


def tokenizar(texto):
    """Lista de palavras do texto, pela REGRA DE OCORRÊNCIA."""
    return _LETRAS.findall(texto.lower())


def padrao_palavra(palavra):
    """Regex que acha a palavra no texto original pela mesma regra (sem letra antes nem depois)."""
    return re.compile(rf"(?<![^\W\d_]){re.escape(palavra)}(?![^\W\d_])", re.IGNORECASE)
