"""Testes de vazamento: o modelo só pode ver o que existia às 21h da véspera."""
import numpy as np
import pandas as pd
import pytest

from pluvio.dataset import resumo_diario, serie_disponivel


def tabela(inicio="2024-10-01 00:00", horas=96):
    idx = pd.date_range(inicio, periods=horas, freq="h", tz="UTC")
    return pd.DataFrame(
        {"precipitation_previous_day1": 1.0, "precipitation_previous_day2": 2.0}, index=idx
    )


def test_conservador_so_usa_day2():
    s = serie_disponivel(tabela(), "precipitation", "conservador")
    assert (s == 2.0).all()


def test_hibrido_troca_para_day2_nas_horas_previstas_depois_da_emissao():
    # dia-alvo local 02/10: emissão às 00:00 UTC de 02/10.
    s = serie_disponivel(tabela(), "precipitation", "hibrido")
    # T = 00:00 UTC de 03/10 -> previsto às 00:00 UTC de 02/10 = emissão: day1
    assert s["2024-10-03 00:00"] == 1.0
    # T = 01:00..03:00 UTC de 03/10 (22h-24h local de 02/10) -> previsto DEPOIS: day2
    assert (s["2024-10-03 01:00":"2024-10-03 03:00"] == 2.0).all()
    # T = 04:00 UTC de 03/10 já é o dia local 03/10 (emissão 00:00 UTC de 03/10): day1 de novo
    assert s["2024-10-03 04:00"] == 1.0


def test_hibrido_nunca_usa_valor_previsto_depois_da_emissao():
    t = tabela(horas=24 * 10)
    s = serie_disponivel(t, "precipitation", "hibrido")
    from pluvio.rotulo import dia_local
    emissao = dia_local(t.index).tz_localize("UTC")          # 00:00 UTC do dia local
    previsto_em = t.index - pd.Timedelta(hours=24)
    usou_day1 = s == 1.0
    assert (previsto_em[usou_day1] <= emissao[usou_day1]).all()


def test_resumo_diario_usa_as_mesmas_janelas_do_rotulo():
    idx = pd.date_range("2024-09-30 04:00", periods=96, freq="h", tz="UTC")
    s = pd.Series(0.0, index=idx)
    s["2024-10-02 03:00"] = 10.0     # 23h-24h local de 01/10
    s["2024-10-02 04:00"] = 10.0     # 00h-01h local de 02/10
    r = resumo_diario(s, "max_3h")
    assert r[pd.Timestamp("2024-10-01")] == 10.0
    assert r[pd.Timestamp("2024-10-02")] == 20.0   # janela cruzando a meia-noite conta para o dia 02


def test_resumo_diario_com_hora_faltando_e_nan():
    idx = pd.date_range("2024-09-30 04:00", periods=96, freq="h", tz="UTC")
    s = pd.Series(0.0, index=idx)
    s["2024-10-02 12:00"] = np.nan
    assert np.isnan(resumo_diario(s, "max_3h")[pd.Timestamp("2024-10-02")])
