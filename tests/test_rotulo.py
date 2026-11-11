import numpy as np
import pandas as pd
import pytest

from pluvio.rotulo import dia_local, rotular_dias, soma_movel


def serie(inicio_utc: str, horas: int, valor: float = 0.0) -> pd.Series:
    idx = pd.date_range(inicio_utc, periods=horas, freq="h", tz="UTC")
    return pd.Series(valor, index=idx)


# O dia local 2024-10-02 (UTC-3, hora = fim do período) são os registros
# de 2024-10-02 04:00 UTC até 2024-10-03 03:00 UTC.
INICIO = "2024-09-30 04:00"   # um dia de folga antes, para o dia 01/10 ter as 24 janelas
DIA2 = pd.Timestamp("2024-10-02")


def test_dia_local_usa_inicio_do_periodo_e_utc_menos_3():
    idx = pd.DatetimeIndex(["2024-10-02 03:00", "2024-10-02 04:00"], tz="UTC")
    assert list(dia_local(idx)) == [pd.Timestamp("2024-10-01"), pd.Timestamp("2024-10-02")]


def test_tres_horas_seguidas_somando_20_e_evento():
    s = serie(INICIO, 96)
    s["2024-10-02 15:00":"2024-10-02 17:00"] = [7.0, 7.0, 6.0]
    assert rotular_dias(s).loc[DIA2, "evento"] == True


def test_19_9_nao_e_evento_e_20_cravado_e():
    s = serie(INICIO, 96)
    s["2024-10-02 15:00"] = 19.9
    assert rotular_dias(s).loc[DIA2, "evento"] == False
    s["2024-10-02 15:00"] = 20.0
    assert rotular_dias(s).loc[DIA2, "evento"] == True


def test_20mm_espalhados_fora_de_3h_seguidas_nao_e_evento():
    s = serie(INICIO, 96)
    s["2024-10-02 10:00"] = 10.0
    s["2024-10-02 13:00"] = 10.0     # 3 horas depois: nunca cabem na mesma janela
    assert rotular_dias(s).loc[DIA2, "evento"] == False


def test_hora_faltando_em_dia_seco_e_desconhecido_nao_falso():
    s = serie(INICIO, 96)
    s["2024-10-02 15:00"] = np.nan
    r = rotular_dias(s)
    assert pd.isna(r.loc[DIA2, "evento"])
    assert r.loc[DIA2, "janelas_ok"] == 21   # uma hora ausente derruba 3 janelas


def test_evento_observado_vale_mesmo_com_buraco_em_outra_hora():
    s = serie(INICIO, 96)
    s["2024-10-02 06:00"] = np.nan
    s["2024-10-02 15:00"] = 25.0
    assert rotular_dias(s).loc[DIA2, "evento"] == True


def test_nan_nao_conta_como_zero_dentro_da_janela():
    s = serie(INICIO, 96)
    s["2024-10-02 15:00"] = 12.0
    s["2024-10-02 16:00"] = np.nan
    s["2024-10-02 17:00"] = 12.0     # 24 mm se o NaN fosse zero
    assert pd.isna(rotular_dias(s).loc[DIA2, "evento"])


def test_janela_que_cruza_a_meia_noite_pertence_ao_dia_em_que_termina():
    s = serie(INICIO, 96)
    # 23h-24h local de 01/10 = registro 03:00 UTC de 02/10; 00h-01h local de 02/10 = 04:00 UTC
    s["2024-10-02 03:00"] = 10.0
    s["2024-10-02 04:00"] = 10.0
    r = rotular_dias(s)
    assert r.loc[pd.Timestamp("2024-10-01"), "evento"] == False   # só 10 mm terminaram no dia 1
    assert r.loc[DIA2, "evento"] == True


def test_soma_movel_recusa_indice_com_buraco():
    s = serie(INICIO, 10).drop(pd.Timestamp("2024-09-30 08:00", tz="UTC"))
    with pytest.raises(ValueError):
        soma_movel(s)
