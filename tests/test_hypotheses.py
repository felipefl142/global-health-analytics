"""Testes da logica de veredito das hipoteses (offline)."""
from src.analysis import hypotheses


def test_veredito_suportada():
    r = hypotheses._resultado("X", "h", "t", -10.0, "u", -12.0, -8.0, 0.001, 100, "-", 5.0)
    assert r["veredito"] == "suportada"
    assert r["significativo"] and r["significancia_pratica"]


def test_veredito_sinal_oposto():
    r = hypotheses._resultado("X", "h", "t", 10.0, "u", 8.0, 12.0, 0.001, 100, "-", 5.0)
    assert r["veredito"] == "rejeitada (sinal oposto)"


def test_veredito_nao_suportada():
    r = hypotheses._resultado("X", "h", "t", -10.0, "u", -12.0, 8.0, 0.4, 100, "-", 5.0)
    assert r["veredito"] == "nao suportada"


def test_veredito_efeito_pequeno():
    r = hypotheses._resultado("X", "h", "t", -1.0, "u", -2.0, 0.0, 0.01, 100, "-", 5.0)
    assert r["veredito"] == "suportada (efeito pequeno)"


def test_veredito_sensivel_a_especificacao():
    """Sinal oposto entre specs significativas => inconclusiva."""
    r = hypotheses._resultado(
        "X", "h", "t", -1.0, "u", -2.0, 0.0, 0.01, 100, "+", 0.5,
        extra={"efeito_entity": 1.5, "p_entity": 0.001},
    )
    assert r["veredito"].startswith("inconclusiva")
