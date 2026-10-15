# Morphogenesis_thesis

Codice della tesi di laurea di Matteo Rizzi (Università degli Studi di Trieste): replica e analisi del modello bioelettrico di Manicka & Levin, nelle varianti *mosaic* e *stigmergica*.

## Uso

Gli script vanno lanciati dalla cartella principale del repository (`python nome_script.py`). I risultati vengono salvati in cartelle create dagli script stessi (`output_rfc/`, `figures/`, …).

I parametri non si passano da riga di comando: sono scritti nel blocco `if __name__ == '__main__':` di ciascuno script (e in alcune costanti in testa al file), quindi per modificarli bisogna aprire e modificare lo script. In alcuni script lo stesso blocco contiene una variabile `MODE` che sceglie quale calcolo eseguire. Fanno eccezione `intervento_saturazione.py` e `robustezza_v.py`, che accettano opzioni da riga di comando.

## Ordine di esecuzione

Alcuni script leggono i risultati di altri, che vanno quindi lanciati prima:

- `grid_and_gp.py` (`MODE = 'grid'`) → `analyze_grid.py`, `grid_estesa.py`, `verifica_criterio_collasso.py`, `fixed_point.py`
- `grid_and_gp.py` (`MODE = 'frontier'`), `refute_rfc.py`, `grid_estesa.py` → `analyze_grid.py` (facoltativi: se mancano, le figure corrispondenti vengono saltate)
- `run_parallel_rfc_and_id.py` (`MODE = 'cloud'`) → `calibration_id.py`
- `train_cfn_gh.py` (`MODE = 'multi_parallel'`, una volta per variante) → `figure_training.py` (i file `face_*_BEST.pt` vanno copiati da `output/` alla cartella principale)
- `timing_map_verifica_S1.py` → `fig_nuove.py` (che richiede anche la cartella `figures/`)

Gli altri script sono indipendenti.

## Contenuto

- `dynamics_t_sparse_gh.py`: dinamica del modello, usata da quasi tutti gli altri script
- `Rfc_and_ID.py`: RFC e dimensione intrinseca della nuvola di stati
- **Replica e addestramento:** `Validation.py`, `train_cfn_gh.py`, `figure_training.py`, `run_autonomous_gh.py`, `run_stimulated_gh.py`, `biforcazione.py`
- **RFC e dimensione intrinseca:** `run_parallel_rfc_and_id.py`, `calibration_id.py`, `mu_test.py`, `refute_rfc.py`, `refute_rfc_timeseries.py`, `rfc_raggio.py`, `id_campi.py`
- **Mappa di fase (W, B):** `grid_and_gp.py`, `analyze_grid.py`, `grid_estesa.py`, `verifica_criterio_collasso.py`, `intervento_saturazione.py`, `robustezza_v.py`, `inversione_di_segno.py`, `campo_max.py`
- **Punti fissi e tempi di convergenza:** `fixed_point.py`, `variation_g.py`, `loop_gain_independence.py`, `timing_map.py`, `timing_map_coda.py`, `timing_map_verifica_S1.py`, `lyap_vblock_all.py`
- **Figure:** `fig_nuove.py`

`MosaicModelParameters.dat` e `StigmergicModelParameters.dat` contengono i parametri dei modelli addestrati pubblicati da Manicka & Levin e sono usati da `Validation.py`.
