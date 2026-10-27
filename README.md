# XAI-RSF

Experimental low-order random survival forest for enterprise first-ST early warning.
Bootstrap survival trees predict leaf Aalen–Johansen CIFs and average them. A path may use at most q distinct input coordinates (q=1,2 or unrestricted). The complete low-order forest is compiled into exact product-reference CIF main and pair functions; no teacher, residual model or post-hoc approximation is fitted.

This is a new forest prototype, **not the existing newRSF2 hazard regression model** and not a bitwise reproduction of GC/scikit-survival or randomForestSRC. Single-cause mode reduces to 1 minus Kaplan–Meier; enterprise mode retains both event types. Company bootstrap, native missing routing and a finite random threshold grid are explicit choices. The GC notebook uses all features per split; the enterprise experiment also uses all 75 input coordinates. No information penalty is claimed in this round.

Install: `python -m pip install -e '.[dev]'`. Test: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q`. Set `CRFOREST_RSCRIPT` and `EW_RLIB` for the two independent R checks.

Provenance: forest core, exact compiler, initial tests and simulation harness imported from the user's newRSF2 prototype commit `2905761d3470c38e8a4be83039dc787f2b07d362`, with subsequent changes tracked here. Existing source workspaces remain unchanged.

Experiments and limitations: `experiments/development_v1/`. This first finite experiment distinguishes unrestricted forest, depth-two forest, path-order-two forest, and additive path-order-one forest. Formal PIT certification is unavailable; enterprise results are PROVISIONAL.
