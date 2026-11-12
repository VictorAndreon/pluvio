"""Transforma o JSON bruto da Open-Meteo em tabelas horárias, uma por modelo.

Nada aqui decide o que o modelo de ML pode ver; só organiza. A regra de
"o que existia às 21h" mora em `dataset.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ler_modelo(pasta: str | Path, modelo: str, estacao: str = "A839") -> pd.DataFrame:
    """Junta os arquivos mensais de um modelo numa tabela horária (índice UTC).

    Os meses foram baixados com 2 dias de sobra de cada lado, então há horas
    repetidas entre arquivos vizinhos. Fica a primeira ocorrência, depois de
    conferir que as repetidas dizem a mesma coisa para as defasagens day1/day2
    (previsão antiga não muda; se mudasse, o arquivo não seria reprodutível).
    """
    arquivos = sorted((Path(pasta) / estacao / modelo).glob("*.json"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum JSON de {modelo} em {pasta}")
    partes = []
    for arquivo in arquivos:
        horario = json.loads(arquivo.read_text(encoding="utf-8"))["resposta"]["hourly"]
        tabela = pd.DataFrame(horario)
        tabela["time"] = pd.to_datetime(tabela["time"], utc=True)
        partes.append(tabela.set_index("time").astype("float64"))
    tudo = pd.concat(partes)
    antigas = [c for c in tudo.columns if "_previous_day" in c]
    # nunique(dropna=False) conta NaN como valor: "tem num arquivo e falta no outro" também é divergência
    versoes = tudo[antigas].groupby(level=0).nunique(dropna=False)
    if (versoes > 1).any().any():
        raise ValueError(f"{modelo}: a mesma hora tem previsões diferentes em arquivos vizinhos")
    unica = tudo[~tudo.index.duplicated(keep="first")].sort_index()
    return unica.asfreq("h")


def grade(pasta: str | Path, modelo: str, estacao: str = "A839") -> dict:
    """Ponto de grade que a API de fato usou para esse modelo."""
    arquivo = sorted((Path(pasta) / estacao / modelo).glob("*.json"))[0]
    r = json.loads(arquivo.read_text(encoding="utf-8"))["resposta"]
    return {"latitude": r["latitude"], "longitude": r["longitude"], "elevacao_m": r["elevation"]}
