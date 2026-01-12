# SelfOpt-MPLC

This repository contains the code developed for the **Self-Optimizing Multi-Plane Light Conversion (MPLC)** project.  
The project focuses on multi-channel optical computing and data-driven optimization techniques.

## Project Structure

- `FLIR_holoeye_active_region_search.py`  
  Implements active region search on the SLM using a FLIR camera and Holoeye SLM.

- `stl_channel_mix_turbo.py`  
  Performs Bayesian optimization–based channel mixing using the TuRBO algorithm.

- `flowers_soc.py`  
  Contains SOC-based optimization helper functions used during the optimization process.

- `flowers_main.py`  
  Main execution script where SOC-based optimization is applied.

## License

MIT License

