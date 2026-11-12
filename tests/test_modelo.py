import numpy as np
import pandas as pd

from pluvio.modelo import dobras, limiar_para_recall, prever_andando


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
    antes = prever_andando(t, ["x"], "2024-06").loc["2024-06"]

    t2 = t.copy()
    t2.loc["2024-07-01":, "x"] = 99.0
    t2.loc["2024-07-01":, "evento"] = True
    depois = prever_andando(t2, ["x"], "2024-06").loc["2024-06"]
    pd.testing.assert_frame_equal(antes, depois)   # prob E alerta (o limiar também não viu o futuro)


def test_limiar_para_recall():
    evento = [1, 1, 1, 1, 0, 0]
    prob = [0.9, 0.6, 0.4, 0.1, 0.8, 0.2]
    assert limiar_para_recall(evento, prob, 0.75) == 0.4   # pega 3 dos 4
    assert limiar_para_recall(evento, prob, 0.50) == 0.6


def test_mesmo_dia_de_duas_estacoes_fica_do_mesmo_lado():
    """Com várias estações, a mesma frente fria aparece em várias linhas do mesmo dia.
    Se uma fosse para o treino e outra para o teste, o modelo colaria."""
    rng = np.random.default_rng(1)
    dias = pd.date_range("2024-01-01", "2024-08-31", freq="D")
    indice = dias.append(dias)                      # duas "estações"
    x = rng.normal(size=len(indice))
    t = pd.DataFrame({"x": x, "evento": x > 1}, index=indice).sort_index()
    saida = prever_andando(t, ["x"], "2024-06")
    assert len(saida) == 2 * len(dias[dias >= "2024-06-01"])
