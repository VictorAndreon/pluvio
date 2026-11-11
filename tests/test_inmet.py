"""Cada teste aqui é uma pegadinha real dos CSVs do INMET."""
import numpy as np
import pandas as pd
import pytest

from pluvio.inmet import ler_csv, ler_pasta

META = "\n".join(f"META{i}:;x" for i in range(8)) + "\n"
ANTIGO = META + (
    "DATA (YYYY-MM-DD);HORA (UTC);PRECIPITACAO;PRESSAO;\n"
    "2015-01-01;00:00;0;931,2;\n"
    "2015-01-01;01:00;-9999;931,9;\n"
    "2015-01-01;02:00;1,4;932,1;\n"
)
NOVO = META + (
    "Data;Hora UTC;PRECIPITACAO;PRESSAO;\n"
    "2019/01/01;0000 UTC;0;934,5;\n"
    "2019/01/01;0100 UTC;;934,4;\n"
    "2019/01/01;0200 UTC;,8;934,4;\n"
)


def escrever(tmp_path, nome, texto):
    caminho = tmp_path / nome
    caminho.write_text(texto, encoding="latin-1")
    return caminho


def test_formato_antigo_menos_9999_vira_nan(tmp_path):
    s = ler_csv(escrever(tmp_path, "a.CSV", ANTIGO))
    assert s.iloc[0] == 0.0
    assert np.isnan(s.iloc[1])          # -9999 não é chuva negativa, é "não sei"
    assert s.iloc[2] == pytest.approx(1.4)   # vírgula decimal


def test_formato_novo_vazio_vira_nan(tmp_path):
    s = ler_csv(escrever(tmp_path, "b.CSV", NOVO))
    assert s.iloc[0] == 0.0
    assert np.isnan(s.iloc[1])          # vazio não é zero
    assert s.iloc[2] == pytest.approx(0.8)   # ",8" sem zero à esquerda


def test_os_dois_formatos_dao_indice_utc(tmp_path):
    antigo = ler_csv(escrever(tmp_path, "a.CSV", ANTIGO))
    novo = ler_csv(escrever(tmp_path, "b.CSV", NOVO))
    assert antigo.index[1] == pd.Timestamp("2015-01-01 01:00", tz="UTC")
    assert novo.index[1] == pd.Timestamp("2019-01-01 01:00", tz="UTC")


def test_hora_sem_linha_no_arquivo_entra_como_nan(tmp_path):
    sem_a_hora_1 = ANTIGO.replace("2015-01-01;01:00;-9999;931,9;\n", "")
    escrever(tmp_path, "a.CSV", sem_a_hora_1)
    s = ler_pasta(tmp_path)
    assert len(s) == 3                  # a hora que faltava foi recriada...
    assert np.isnan(s.iloc[1])          # ...como "não sei"
