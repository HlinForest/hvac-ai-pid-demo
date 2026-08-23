# OpenModelica physical-reference layer

This package is the traceable physical reference for the Python 3R2C/FOPDT
controller experiments. The executable reference uses components from the
Modelica Standard Library so it can run immediately after OpenModelica is
installed. The same topology can later be replaced by higher-fidelity
Modelica Buildings components without changing the Python control interface.
The zone/wall capacities, three thermal resistances, cooling capacity, delay,
actuator time constant, weather and finite door pulse mirror the Python model.

Run the complete compile, 12-hour simulation, cross-validation and report
refresh workflow from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File modelica\run_openmodelica_validation.ps1
```

If the current terminal is not in the project root, use the absolute path or
double-click `run_openmodelica_validation.bat` in the project root. The batch
wrapper resolves its location with `%~dp0`, so it does not depend on the
terminal's current directory.

To inspect and rerun the model interactively in OMEdit, close any existing
OMEdit instance and double-click `open_modelica_gui.bat`. See
`modelica/OMEDIT_GUI_GUIDE.md` for the exact variables to plot.

The wrapper deliberately uses one compiler process and disables parallel code
generation. This avoids the Clang frontend crashes observed during parallel
compilation on this machine. It also creates a temporary ASCII-only `M:` drive
mapping because the OpenModelica package manager cannot reliably decode this
machine's Chinese user-profile path; the mapping is removed on exit.

1. Install OpenModelica (the included Modelica Standard Library is sufficient).
2. Add this `modelica` directory to the Modelica path.
3. Simulate `HVACAI.PrecisionCabinetCooling` for 43,200 seconds.
4. Apply a change to `command` and export the zone-temperature response; use
   that response to compare with `hvac_pid.controllers.identify_fopdt`.
5. Export CSV with `time`, `zone.T`, and `commandInput.y`, then run:

   ```powershell
   python extract_fopdt.py PrecisionCabinetCooling_res.csv --kelvin
   ```

   The script automatically produces the process gain, time constant, and
   delay used to cross-check the fast FOPDT proxy.

The Python demo deliberately remains independent of an OpenModelica runtime so
that batch optimization and the Streamlit demo can execute on development PCs
and CI machines without a Modelica installation.

`hvac_pid.validation.run_cross_validation` always runs two numerical checks:
the 1-minute discrete model against an independent SciPy DOP853 solution and
the identified FOPDT proxy against the 3R2C step response. When the Modelica
result CSV exists, it also compares the actual DASSL result with a matching
independent DOP853 continuous-equation run and exports the third evidence set.
