# pluvio

Benchmark de previsão de chuva forte em Passo Fundo/RS e um modelo simples de
ML que tenta combinar as previsões dos modelos numéricos.

> **Estado:** em construção. A seção de resultados ainda não foi escrita; o
> período de teste (jul–out/2026) continua trancado.

## A pergunta

Às 21h (horário local) de hoje: **amanhã vai ter alguma janela de 3 horas
seguidas com 20 mm ou mais de chuva?** Sim ou não.

- O ponto de partida foi o aviso amarelo do INMET (20 mm em 1 hora). Na estação
  de Passo Fundo isso aconteceu só 2 vezes em 10 meses de 2026, pouco demais
  para medir qualquer coisa. O limiar foi adaptado para 20 mm em 3 horas.
- Esse limiar foi escolhido olhando a frequência de eventos em 2026, o que
  inclui os meses de teste. A escolha foi por frequência, não por desempenho de
  modelo, mas fica registrada.
- Perder um temporal é pior que dar um alarme falso, então a métrica principal
  fixa o recall e pergunta quantos alarmes falsos isso custa.

## Dados

| O quê | Fonte | Período |
|---|---|---|
| Chuva observada (a verdade) | Estação automática INMET A839, CSVs anuais de `portal.inmet.gov.br/dadoshistoricos` | nov/2006 – out/2026 |
| Previsões dos modelos | [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | fev/2024 – out/2026 |

Modelos comparados: `gfs_global` (NOAA), `icon_global` (DWD) e `ecmwf_ifs025`
(ECMWF). Os dados brutos estão em `data/raw/`, exatamente como foram baixados.

## Decisões de método

Cada item abaixo muda os números se for feito de outro jeito.

### Rótulo (`src/pluvio/rotulo.py`)

1. **Hora = fim do período.** O registro das 15:00 UTC é a chuva entre 14:00 e
   15:00. É a convenção documentada da Open-Meteo e a premissa adotada para o
   INMET.
2. **Dia local = UTC−3 fixo.** Passo Fundo teve horário de verão até 2019; com
   o relógio de parede, dois dias por ano teriam 23h ou 25h.
3. **A janela pertence ao dia em que termina.** Um temporal das 23h à 1h conta
   para o dia seguinte, em vez de sumir na virada.
4. **Hora faltando é "não sei", nunca zero.** O dia vale *evento* se alguma
   janela completa atingiu 20 mm; vale *sem evento* só se as 24 janelas estão
   completas; caso contrário fica fora da conta.

### O que o modelo pode ver (`src/pluvio/dataset.py`)

A previsão é emitida às 21h da véspera. O sufixo `_previous_day1` da Open-Meteo
significa "valor previsto 24 horas antes de cada hora-alvo", e não "o que
existia às 21h". Para as últimas 3 horas do dia-alvo, 24 horas antes já é
depois da emissão. Além disso, a documentação não diz se "previsto 24h antes"
conta pela hora em que a rodada foi iniciada ou publicada.

Por isso há dois modos, e os dois são reportados:

| Modo | Usa | Leitura |
|---|---|---|
| `conservador` | `_previous_day2` em todas as horas | Sem vazamento sob qualquer interpretação. É um piso: descarta a rodada mais recente. |
| `hibrido` | `_previous_day1` até onde 24h antes ≤ emissão, `_previous_day2` depois | Mais perto do uso real, mas depende da interpretação da API. É um teto otimista. |

Outros cuidados:

- **Persistência honesta.** "Choveu forte hoje?" usa só o que foi observado até
  as 21h, não o dia inteiro.
- **Climatologia do passado.** A frequência de eventos por mês usa só
  2006–2023, antes de qualquer dia que o modelo tenta prever.
- **Mesmas janelas para previsão e observação.** A chuva prevista é resumida
  como a maior soma de 3h entre as mesmas 24 janelas do rótulo.

### Avaliação (`src/pluvio/avaliacao.py`, `src/pluvio/modelo.py`)

- **Acurácia não é usada.** Com ~5% de dias positivos, prever "nunca" acerta
  95% dos dias e nenhum temporal.
- **Janela expansível.** Cada mês é previsto por um modelo treinado só com os
  meses anteriores. Sorteio aleatório de dias vazaria, porque dias vizinhos
  pertencem ao mesmo sistema de tempo.
- **Nada é ajustado olhando resultado.** Regressão logística com `C=1.0`,
  padronização aprendida só no treino, sem busca de hiperparâmetro.
- **Teste trancado.** `benchmark.py` corta a tabela em 30/06/2026 antes de
  calcular qualquer coisa.

## Limitações conhecidas

- **Poucos eventos.** São cerca de 40 no período com previsões arquivadas. Os
  intervalos de confiança são largos e estão nas tabelas.
- **Ponto contra grade.** A estação mede um ponto; os modelos preveem a média
  de uma célula de 13 a 28 km. Um temporal localizado aparece diluído na
  previsão, então o limiar de 20 mm não vale direto para os modelos.
- **ECMWF em blocos de 3h.** A Open-Meteo repete o mesmo valor nas 3 horas do
  bloco. Os totais mensais batem com os dos outros modelos, o que indica que o
  valor já é o total do bloco dividido por 3.
- **Falhas da estação.** 21% das horas de 2025 e 11% das de 2026 estão
  ausentes.
- **2024 foi um ano anômalo** (El Niño, enchentes de maio no RS) e concentra a
  maior parte dos eventos.

## Como rodar

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

```bash
python -m pluvio.coleta --ate 2026-10
```

```bash
python -m pluvio.benchmark --modo conservador
```

## Estrutura

```
data/raw/inmet/        CSVs da estação, como vieram do INMET
data/raw/openmeteo/    JSON bruto da API, um por modelo e mês
src/pluvio/inmet.py    leitura dos dois formatos de CSV
src/pluvio/rotulo.py   janela de 3h e rótulo diário
src/pluvio/coleta.py   download das previsões
src/pluvio/previsoes.py  JSON bruto -> tabela horária
src/pluvio/dataset.py  tabela diária sem vazamento
src/pluvio/avaliacao.py  métricas
src/pluvio/modelo.py   regressão logística em janela expansível
src/pluvio/benchmark.py  roda tudo no período de desenvolvimento
tests/                 um teste por pegadinha
```

## Créditos dos dados

Observações: Instituto Nacional de Meteorologia (INMET). Previsões:
[Open-Meteo](https://open-meteo.com/), a partir de dados de NOAA, DWD e ECMWF.
