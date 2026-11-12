"""Coleta das previsões na Previous Runs API da Open-Meteo.

Regra de ouro: este script só BAIXA e GUARDA. Nenhuma conta, nenhum filtro.
O JSON vai para o disco exatamente como chegou, junto com os parâmetros do
pedido e a hora do download. Se um dia descobrirmos um erro de interpretação
(fuso, bloco de 3h, sufixo), reprocessamos o bruto sem depender da API devolver
a mesma coisa de novo.

Uso:
    python -m pluvio.coleta --ate 2026-10  # de jan/2024 até o mês indicado (inclusive)

Um arquivo por modelo e por mês: data/raw/openmeteo/<modelo>/<AAAA-MM>.json
Arquivo que já existe não é baixado de novo.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LATITUDE, LONGITUDE = -28.2267, -52.4036   # estação INMET A839

# Modelos globais explícitos (não os "_seamless", que podem trocar de modelo
# por baixo dos panos ao longo do tempo).
MODELOS = ["gfs_global", "icon_global", "ecmwf_ifs025"]
VARIAVEIS = ["precipitation", "cape"]
# day0 = rodada mais recente (referência, NÃO serve como entrada do modelo);
# day1/day2 = previsto 24h/48h antes da hora-alvo.
DEFASAGENS = ["", "_previous_day1", "_previous_day2"]

PRIMEIRO_MES = "2024-01"
# O dia local (UTC-3) transborda para o dia UTC seguinte, e a janela de 3h
# olha para trás: pedimos 2 dias de sobra de cada lado de cada mês.
MARGEM_DIAS = 2
PAUSA_S = 2.0


def parametros(modelo: str, mes: pd.Period) -> dict:
    margem = pd.Timedelta(days=MARGEM_DIAS)
    return {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(v + d for v in VARIAVEIS for d in DEFASAGENS),
        "models": modelo,
        "start_date": (mes.start_time - margem).strftime("%Y-%m-%d"),
        "end_date": (mes.end_time + margem).strftime("%Y-%m-%d"),
        "timezone": "GMT",   # explícito: tudo em UTC, igual ao INMET
    }


def pedir_com_paciencia(params: dict, tentativas: int = 15) -> dict:
    """GET que devolve o JSON já conferido, insistindo em falhas passageiras.

    Repete (esperando 5s, 10s, 20s, depois 30s) quando:
      - a rede cai;
      - vem HTTP 429 ("devagar aí": a API gratuita limita pedidos por IP) ou 5xx;
      - o corpo não é JSON ou não tem o bloco "hourly".
    Erro 4xx diferente de 429 é pedido errado; repetir não adianta, estoura na hora.
    """
    espera = 5.0
    for tentativa in range(1, tentativas + 1):
        try:
            resposta = requests.get(URL, params=params, timeout=120)
            if 400 <= resposta.status_code < 500 and resposta.status_code != 429:
                raise RuntimeError(f"pedido recusado: HTTP {resposta.status_code} {resposta.text[:300]}")
            dados = resposta.json() if resposta.status_code == 200 else None
            if isinstance(dados, dict) and "hourly" in dados:
                return dados
            motivo = f"HTTP {resposta.status_code} sem dados"
        except (requests.ConnectionError, requests.Timeout, ValueError) as erro:
            motivo = type(erro).__name__
        if tentativa == tentativas:
            raise RuntimeError(f"desisti depois de {tentativas} tentativas ({motivo})")
        print(f"  {motivo}; esperando {espera:.0f}s", flush=True)
        time.sleep(espera)
        espera = min(espera * 2, 30.0)


def baixar(modelo: str, mes: pd.Period, pasta: Path) -> str:
    destino = pasta / modelo / f"{mes}.json"
    if destino.exists():
        return "já existe"
    params = parametros(modelo, mes)
    dados = pedir_com_paciencia(params)
    pacote = {
        "pedido": {"url": URL, "params": params},
        "baixado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resposta": dados,
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    # escreve num temporário e renomeia: download interrompido não deixa
    # arquivo pela metade com cara de completo
    temporario = destino.with_suffix(".parcial")
    temporario.write_text(json.dumps(pacote, ensure_ascii=False), encoding="utf-8")
    temporario.rename(destino)
    time.sleep(PAUSA_S)
    return "baixado"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--de", default=PRIMEIRO_MES, help="primeiro mês, AAAA-MM")
    ap.add_argument("--ate", required=True, help="último mês, AAAA-MM (inclusive)")
    ap.add_argument("--pasta", default="data/raw/openmeteo")
    args = ap.parse_args()

    for mes in pd.period_range(args.de, args.ate, freq="M"):
        for modelo in MODELOS:
            print(f"{mes} {modelo}: {baixar(modelo, mes, Path(args.pasta))}", flush=True)


if __name__ == "__main__":
    main()
