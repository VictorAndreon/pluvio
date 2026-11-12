import numpy as np
import pytest

from pluvio.avaliacao import (brier, brier_skill, contingencia, curva_de_limiares,
                              ponto_de_operacao, precisao_media, wilson)


def test_contingencia_na_mao():
    evento = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    alerta = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    c = contingencia(evento, alerta)
    assert (c["acertos"], c["alarmes_falsos"], c["perdas"], c["negativos"]) == (2, 1, 1, 6)
    assert c["recall"] == pytest.approx(2 / 3)
    assert c["precisao"] == pytest.approx(2 / 3)
    assert c["csi"] == pytest.approx(2 / 4)


def test_nunca_alertar_tem_acuracia_alta_e_recall_zero():
    evento = [1] + [0] * 19          # 5% de eventos
    c = contingencia(evento, [0] * 20)
    assert (c["acertos"] + c["negativos"]) / 20 == 0.95
    assert c["recall"] == 0.0


def test_wilson_8_de_12():
    baixo, alto = wilson(8, 12)
    assert baixo == pytest.approx(0.39, abs=0.01)
    assert alto == pytest.approx(0.86, abs=0.01)


def test_brier_e_skill():
    evento = [1, 0, 0, 0]
    assert brier(evento, [1, 0, 0, 0]) == 0.0
    assert brier(evento, [0.25] * 4) == pytest.approx(0.1875)      # climatologia
    assert brier_skill(evento, [1, 0, 0, 0], [0.25] * 4) == 1.0
    assert brier_skill(evento, [0.25] * 4, [0.25] * 4) == 0.0


def test_precisao_media_perfeita_e_pior_caso():
    assert precisao_media([1, 1, 0, 0], [9, 8, 1, 0]) == 1.0
    # eventos no fim da fila: precisões 1/3 e 2/4
    assert precisao_media([0, 0, 1, 1], [9, 8, 1, 0]) == pytest.approx((1 / 3 + 2 / 4) / 2)


def test_ponto_de_operacao_escolhe_o_limiar_mais_alto_que_atinge_o_recall():
    evento = [1, 0, 1, 0, 0]
    mm = [30, 25, 10, 5, 0]
    p = ponto_de_operacao(evento, mm, recall_minimo=1.0)
    assert p["limiar"] == 10
    assert p["alarmes_falsos"] == 1
    assert ponto_de_operacao(evento, mm, recall_minimo=0.5)["limiar"] == 30
