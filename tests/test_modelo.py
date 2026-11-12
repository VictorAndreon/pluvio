import numpy as np
import pandas as pd

from pluvio.modelo import dobras, prever_andando


def test_treino_fica_sempre_antes_do_mes_previsto():
    dias = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    n = 0
    for treino, previsto in dobras(dias, "2024-06"):
        assert treino.max() < previsto.min()
        assert previsto.to_period("M").nunique() == 1
        n += 1
    assert n == 7   # jun..dez


def test_previsao_de_um_mes_nao_muda_se_o_futuro_mudar():
    """Se mexer nos dados de julho alterasse a previsão de junho, haveria vazamento."""
    rng = np.random.default_rng(0)
    dias = pd.date_range("2024-01-01", "2024-08-31", freq="D")
    x = rng.normal(size=len(dias))
    t = pd.DataFrame({"x": x, "evento": x + rng.normal(size=len(dias)) > 1}, index=dias)
    antes = prever_andando(t, ["x"], "2024-06")["2024-06"]

    t2 = t.copy()
    t2.loc["2024-07-01":, "x"] = 99.0
    t2.loc["2024-07-01":, "evento"] = True
    depois = prever_andando(t2, ["x"], "2024-06")["2024-06"]
    pd.testing.assert_series_equal(antes, depois)
