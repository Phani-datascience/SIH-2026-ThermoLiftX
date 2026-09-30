# ThermoLift-X FINAL v3.0

Complete runnable prototype for SIH PS 26120.

## Included
- Reservoir thermal state
- Thermal Memory Index
- Wellbore pressure/temperature/viscosity state
- SRP pump fillage and rod-load intelligence
- Rod floating risk
- Pump unsetting risk
- Production, SOR and energy prediction surrogates
- Failure-risk classifier
- Coupled CSS + SRP counterfactual optimizer
- Explicit safety envelope
- Counterfactual search landscape
- Explainable recommendation
- 30-day future simulator
- Multi-well selector
- Model/data validation page

## Run
Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell activation is blocked, use CMD:
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the localhost address Streamlit prints, normally http://localhost:8501.

## Demo order
1. Executive Twin
2. CSS + SRP Optimizer → RUN OPTIMIZER
3. SRP Intelligence
4. Future Simulator
5. Data + Models

## Important
The data and engineering limits are synthetic/prototype values. They are not Oil India/Baghewala measurements and must not be used for field decisions. A real deployment requires authorized field data, PVT data, completion information, dynamometer cards, validated operating envelopes and petroleum-engineering review.
