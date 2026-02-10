# Dev-contitional Platform

Hlavný repozitár je rozdelený na 2 prepojené, ale samostatne prezentované projekty:

- **60%: Krystal Vino + GAMESA 3D Grid** (hlavná platforma)
- **40%: FANUC RISE** (vedľajšia priemyselná vetva, FOCAS integrácia)

---

## 1) Krystal Vino + GAMESA 3D Grid (hlavná platforma, 60%)

### Čo je Krystal Vino
Krystal Vino je výkonová orchestrace nad OpenVINO/oneAPI pre osobné počítače.  
Cieľom je znižovať latenciu a zvyšovať throughput pomocou adaptívneho plánovania, telemetrie a riadenia runtime politík.

Kód: `openvino_oneapi_system/`

### Kľúčové komponenty
- **OpenVINO runtime vrstva**: inferencia s fallback režimom.
- **oneAPI/OpenMP tuning**: dynamické nastavovanie `ONEAPI_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENVINO_NUM_STREAMS`, `KMP_*`.
- **Ekonomický planner + evolučný tuner**: online voľba režimov `defensive/balanced/aggressive`.
- **GAMESA 3D Grid**: logická 3D pamäťová vrstva pre organizáciu/swap pracovných dát.
- **Delegované logovanie**: samostatné kanály `system`, `telemetry`, `planning`, `policy`, `inference`, `grid_update`.

### Preukázateľné výsledky (Linux benchmark)
Zdroj: `openvino_oneapi_system/logs/benchmark_latest.txt`

- **Latency improvement**: `66.01%`
- **Throughput improvement**: `234.59%`
- **Utility improvement**: `270.42%`
- **Sysbench improvement**: `99.55%`  
  Baseline: `2615.43 events/s` -> Adaptive: `5219.10 events/s`

### Rýchle spustenie
```bash
python3 openvino_oneapi_system/main.py --cycles 10 --interval 0.5
python3 openvino_oneapi_system/benchmark_linux.py --cycles 60
```

### Debian balík (whole package)
Vygenerovaný balík:
- `openvino_oneapi_system/dist/openvino-oneapi-system_1.1.0_amd64.deb`

Obsahuje:
- CLI: `ovo-runtime`, `ovo-benchmark`
- service unit: `openvino-oneapi-system.service`
- konfig: `/etc/default/openvino-oneapi-system`

---

## 2) FANUC RISE (sekundárna vetva, 40%)

### Charakteristika projektu
FANUC RISE je priemyselná CNC vrstva orientovaná na operácie, telemetry a workflow automatizáciu.  
FOCAS je tu **vedľajšia integračná vrstva**, nie hlavný produktový cieľ.

Kód: `advanced_cnc_copilot/`

### Zameranie
- CNC operátorské workflow a dohľad
- API + UI pre výrobný monitoring
- FANUC telemetry bridge (mock/real režim podľa prostredia)
- rozšíriteľné backend služby pre výrobnú analytiku

### Kde dáva zmysel v celom ekosystéme
- Krystal Vino rieši výkonový runtime a optimalizáciu výpočtu.
- FANUC RISE rieši priemyselný kontext, machine/data napojenie a operátorské použitie.
- Spolu tvoria pipeline: **výkonové jadro + priemyselná exekúcia**.

---

## Repo mapa
- `openvino_oneapi_system/` hlavná výkonnostná platforma (OpenVINO, oneAPI, GAMESA 3D Grid)
- `advanced_cnc_copilot/` FANUC RISE priemyselný stack
- `docs/` doplnkové technické podklady

## Poznámka k smerovaniu
Priorita repozitára je Krystal Vino/GAMESA 3D Grid ako hlavná platforma pre PC hardvér a inferenčný výkon.  
FANUC RISE zostáva samostatná, sekundárna doménová vetva pre CNC integrácie.
