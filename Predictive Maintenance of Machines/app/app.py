from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# The trained model is stored in the repository's model folder.
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "failure_type_model.joblib"

_model = None


def get_model():
    """Load the model once and cache it, instead of reloading on every request."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Place failure_type_model.joblib in the model folder, or edit "
                "MODEL_PATH at the top of this file."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# Maps clean JSON keys (used by the frontend) to the exact column names the
# model's ColumnTransformer was fitted on. A couple of the real column names
# have units in brackets or a trailing space, which are awkward to use as
# JSON keys directly, so we translate here instead.
FIELD_MAP = {
    "airTemperature": "Air temperature [K]",
    "processTemperature": "Process temperature [K]",
    "rotationalSpeed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "vibrationLevels": "Vibration Levels ",
    "operationalHours": "Operational Hours",
    "power": "Power [kW]",
}

VALID_TYPES = ("L", "M", "H")


def build_input_frame(payload):
    """Validate a JSON payload and turn it into the one-row DataFrame the
    model's ColumnTransformer expects."""
    machine_type = payload.get("type")
    if machine_type not in VALID_TYPES:
        raise ValueError("'type' must be one of 'L', 'M', 'H'.")

    row = {"Type": [machine_type]}
    for json_key, column_name in FIELD_MAP.items():
        if json_key not in payload or payload[json_key] in (None, ""):
            raise ValueError(f"Missing field: '{json_key}'.")
        try:
            row[column_name] = [float(payload[json_key])]
        except (TypeError, ValueError):
            raise ValueError(f"'{json_key}' must be a number.")

    return pd.DataFrame(row)


# Note: this model's RBF kernel (gamma=0.1) means predictions can go flat
# ("No Failure" with an identical decision_function) once an input drifts
# even modestly outside the training distribution -- the kernel similarity
# to every support vector collapses toward zero. Realistic sensor values
# matter more than usual here; see the placeholder text in each field for
# ranges pulled from this model's own support vectors.
def score_classes(model, frame):
    """Turn the SVC's decision_function output into a relative signal
    strength per class. This is NOT a calibrated probability (the model was
    trained with probability=False) -- it's a softmax over the one-vs-rest
    margins, useful only for showing which classes were close calls."""
    raw = model.decision_function(frame)[0]
    shifted = raw - raw.max()
    weights = np.exp(shifted)
    softmax = weights / weights.sum()
    return {
        cls: round(float(score), 4)
        for cls, score in sorted(
            zip(model.classes_, softmax), key=lambda pair: pair[1], reverse=True
        )
    }


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Failure-Type Diagnostic Panel</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --navy-950: #0a1826;
    --navy-800: #10263b;
    --navy-700: #16324c;
    --line: #3e7cb1;
    --line-dim: #234a68;
    --bright: #8ecbf0;
    --paper: #eee8da;
    --paper-dim: #d9d2c1;
    --ink: #0f1d2b;
    --ink-muted: #5b6b7a;
    --safe: #4f9a6b;
    --safe-dim: #2c4536;
    --alarm: #e0a52c;
    --alarm-dim: #4a3a1c;
    --danger: #c9503f;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
    color: var(--paper);
    background-color: var(--navy-950);
    background-image:
      linear-gradient(var(--line-dim) 1px, transparent 1px),
      linear-gradient(90deg, var(--line-dim) 1px, transparent 1px);
    background-size: 28px 28px;
    background-position: -1px -1px;
    min-height: 100vh;
    padding: 56px 20px 80px;
  }

  h1, h2, .display {
    font-family: 'Space Grotesk', sans-serif;
  }

  .wrap {
    max-width: 780px;
    margin: 0 auto;
  }

  .masthead {
    display: flex;
    align-items: flex-start;
    gap: 18px;
    margin-bottom: 8px;
  }

  .gauge-mark {
    flex: none;
    width: 44px;
    height: 44px;
    margin-top: 4px;
  }

  .masthead h1 {
    font-size: 27px;
    font-weight: 700;
    color: #fdfcf9;
    letter-spacing: -0.01em;
    margin: 0 0 6px;
  }

  .masthead p {
    margin: 0;
    color: var(--bright);
    font-size: 14px;
    line-height: 1.6;
    max-width: 46ch;
  }

  .tag-line {
    margin: 18px 0 34px;
    padding-top: 14px;
    border-top: 1px dashed var(--line-dim);
    font-size: 12px;
    color: var(--ink-muted);
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
  }

  .tag-line b { color: var(--bright); font-weight: 500; }

  .panel {
    position: relative;
    background: var(--navy-800);
    border: 1px solid var(--line-dim);
    padding: 36px 38px 32px;
  }

  .panel::before,
  .panel::after,
  .panel .corner-tl,
  .panel .corner-br {
    content: '';
    position: absolute;
    width: 16px;
    height: 16px;
    border: 2px solid var(--bright);
  }
  .panel::before { top: -1px; left: -1px; border-right: none; border-bottom: none; }
  .panel::after { bottom: -1px; right: -1px; border-left: none; border-top: none; }

  .section {
    margin-bottom: 30px;
  }
  .section:last-of-type { margin-bottom: 0; }

  .section-label {
    font-size: 12px;
    color: var(--bright);
    letter-spacing: 0.02em;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px dashed var(--line-dim);
  }

  .type-row {
    display: flex;
    gap: 10px;
  }

  .type-option {
    flex: 1;
    position: relative;
  }

  .type-option input {
    position: absolute;
    opacity: 0;
    inset: 0;
    cursor: pointer;
    margin: 0;
  }

  .type-option label {
    display: block;
    text-align: center;
    padding: 12px 8px 10px;
    background: var(--navy-700);
    border: 1px solid var(--line-dim);
    color: var(--paper-dim);
    font-size: 13px;
    cursor: pointer;
    transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
  }

  .type-option label .code {
    display: block;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 19px;
    font-weight: 600;
    margin-bottom: 2px;
  }

  .type-option label .name {
    font-size: 10px;
    color: var(--ink-muted);
  }

  .type-option input:checked + label {
    border-color: var(--bright);
    background: var(--navy-950);
    color: var(--bright);
  }

  .type-option input:checked + label .name { color: var(--bright); opacity: 0.8; }

  .type-option input:focus-visible + label {
    outline: 2px solid var(--bright);
    outline-offset: 2px;
  }

  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px 22px;
  }

  .field label {
    display: block;
    font-size: 12px;
    color: var(--paper-dim);
    margin-bottom: 7px;
  }

  .field-input {
    display: flex;
    align-items: stretch;
    border: 1px solid var(--line-dim);
    background: var(--paper);
  }

  .field-input input {
    flex: 1;
    min-width: 0;
    border: none;
    background: transparent;
    color: var(--ink);
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    padding: 10px 12px;
  }

  .field-input input:focus {
    outline: none;
  }

  .field-input:focus-within {
    border-color: var(--bright);
  }

  .field-input .unit {
    flex: none;
    display: flex;
    align-items: center;
    padding: 0 12px;
    background: var(--paper-dim);
    color: var(--ink-muted);
    font-size: 11px;
  }

  input[type=number]::-webkit-outer-spin-button,
  input[type=number]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  input[type=number] { -moz-appearance: textfield; }

  .run-row {
    margin-top: 36px;
    display: flex;
    align-items: center;
    gap: 16px;
  }

  button#runBtn {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 14px;
    color: var(--navy-950);
    background: var(--bright);
    border: none;
    padding: 13px 26px;
    cursor: pointer;
    transition: background 0.15s ease, transform 0.05s ease;
  }

  button#runBtn:hover { background: #a9dbf7; }
  button#runBtn:active { transform: translateY(1px); }
  button#runBtn:disabled { background: var(--line-dim); color: var(--ink-muted); cursor: default; }

  .run-hint {
    font-size: 11px;
    color: var(--ink-muted);
  }

  .readout {
    margin-top: 40px;
    border-top: 1px solid var(--line-dim);
    padding-top: 26px;
    display: none;
  }

  .readout.visible { display: block; }

  .readout-label {
    font-size: 12px;
    color: var(--bright);
    margin-bottom: 14px;
  }

  .status-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 22px;
  }

  .led {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--ink-muted);
    box-shadow: 0 0 0 3px rgba(255,255,255,0.04);
    flex: none;
  }

  .led.safe { background: var(--safe); box-shadow: 0 0 10px 1px var(--safe); }
  .led.alarm { background: var(--alarm); box-shadow: 0 0 10px 1px var(--alarm); }
  .led.error { background: var(--danger); box-shadow: 0 0 10px 1px var(--danger); }

  .status-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: #fdfcf9;
  }

  .status-text .qualifier {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 400;
    color: var(--ink-muted);
    display: block;
    margin-top: 3px;
  }

  .bars { display: flex; flex-direction: column; gap: 10px; }

  .bar-row {
    display: grid;
    grid-template-columns: 140px 1fr 48px;
    align-items: center;
    gap: 12px;
    font-size: 11px;
    color: var(--paper-dim);
  }

  .bar-track {
    height: 6px;
    background: var(--navy-700);
    position: relative;
    overflow: hidden;
  }

  .bar-fill {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 0%;
    background: var(--line-dim);
    transition: width 0.4s ease;
  }

  .bar-row.top .bar-fill { background: var(--bright); }
  .bar-row.top { color: var(--bright); }

  .bar-value { text-align: right; color: var(--ink-muted); }

  .disclaimer {
    margin-top: 18px;
    font-size: 10.5px;
    color: var(--ink-muted);
    line-height: 1.5;
  }

  @media (max-width: 620px) {
    .grid-2 { grid-template-columns: 1fr; }
    .panel { padding: 28px 20px; }
    .bar-row { grid-template-columns: 100px 1fr 40px; }
  }
</style>
</head>
<body>
<div class="wrap">

  <div class="masthead">
    <svg class="gauge-mark" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="24" r="19" stroke="#8ecbf0" stroke-width="2"/>
      <path d="M24 24 L24 10" stroke="#8ecbf0" stroke-width="2" stroke-linecap="round"/>
      <path d="M24 24 L33 30" stroke="#e0a52c" stroke-width="2" stroke-linecap="round"/>
      <circle cx="24" cy="24" r="2.5" fill="#8ecbf0"/>
    </svg>
    <div>
      <h1>Failure-type diagnostic panel</h1>
      <p>Enter a machine's live sensor readings to classify the type of failure it is heading toward, if any.</p>
    </div>
  </div>

  <div class="tag-line">
    <span><b>Model</b> SVC (RBF), oversampled training set</span>
    <span><b>Classes</b> No Failure · Overstrain · Power · Tool Wear</span>
  </div>

  <form class="panel" id="diagForm">
    <div class="corner-tl"></div>

    <div class="section">
      <div class="section-label">Product type</div>
      <div class="type-row">
        <div class="type-option">
          <input type="radio" name="type" id="type-l" value="L" checked>
          <label for="type-l"><span class="code">L</span><span class="name">Low</span></label>
        </div>
        <div class="type-option">
          <input type="radio" name="type" id="type-m" value="M">
          <label for="type-m"><span class="code">M</span><span class="name">Medium</span></label>
        </div>
        <div class="type-option">
          <input type="radio" name="type" id="type-h" value="H">
          <label for="type-h"><span class="code">H</span><span class="name">High</span></label>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-label">Thermal &amp; mechanical readings</div>
      <div class="grid-2">
        <div class="field">
          <label for="airTemperature">Air temperature</label>
          <div class="field-input">
            <input type="number" step="any" id="airTemperature" placeholder="298.1" required>
            <span class="unit">K</span>
          </div>
        </div>
        <div class="field">
          <label for="processTemperature">Process temperature</label>
          <div class="field-input">
            <input type="number" step="any" id="processTemperature" placeholder="308.3" required>
            <span class="unit">K</span>
          </div>
        </div>
        <div class="field">
          <label for="rotationalSpeed">Rotational speed</label>
          <div class="field-input">
            <input type="number" step="any" id="rotationalSpeed" placeholder="1500" required>
            <span class="unit">rpm</span>
          </div>
        </div>
        <div class="field">
          <label for="torque">Torque</label>
          <div class="field-input">
            <input type="number" step="any" id="torque" placeholder="40.2" required>
            <span class="unit">Nm</span>
          </div>
        </div>
        <div class="field">
          <label for="vibrationLevels">Vibration level</label>
          <div class="field-input">
            <input type="number" step="any" id="vibrationLevels" placeholder="34" required>
          </div>
        </div>
        <div class="field">
          <label for="power">Power</label>
          <div class="field-input">
            <input type="number" step="any" id="power" placeholder="5.9" required>
            <span class="unit">kW</span>
          </div>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-label">Operational</div>
      <div class="grid-2">
        <div class="field">
          <label for="operationalHours">Operational hours</label>
          <div class="field-input">
            <input type="number" step="any" id="operationalHours" placeholder="85" required>
            <span class="unit">hrs</span>
          </div>
        </div>
      </div>
    </div>

    <div class="run-row">
      <button type="submit" id="runBtn">Run diagnostic</button>
      <span class="run-hint" id="runHint">All fields required</span>
    </div>

    <div class="readout" id="readout">
      <div class="readout-label">Status readout</div>
      <div class="status-row">
        <div class="led" id="led"></div>
        <div class="status-text" id="statusText">--</div>
      </div>
      <div class="bars" id="bars"></div>
      <div class="disclaimer">Bars show relative signal strength across classes (softmax over the classifier's decision margins) &mdash; not a calibrated probability, since the model was trained without probability estimation.</div>
    </div>

    <div class="corner-br"></div>
  </form>

</div>

<script>
const form = document.getElementById('diagForm');
const runBtn = document.getElementById('runBtn');
const runHint = document.getElementById('runHint');
const readout = document.getElementById('readout');
const led = document.getElementById('led');
const statusText = document.getElementById('statusText');
const bars = document.getElementById('bars');

const NUMERIC_IDS = [
  'airTemperature', 'processTemperature', 'rotationalSpeed',
  'torque', 'vibrationLevels', 'power', 'operationalHours'
];

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const payload = { type: form.querySelector('input[name="type"]:checked').value };
  for (const id of NUMERIC_IDS) {
    payload[id] = document.getElementById(id).value;
  }

  runBtn.disabled = true;
  runHint.textContent = 'Running...';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    if (!res.ok) {
      throw new Error(result.error || 'Prediction request failed.');
    }

    renderResult(result);
  } catch (err) {
    renderError(err.message);
  } finally {
    runBtn.disabled = false;
    runHint.textContent = 'All fields required';
  }
});

function renderResult(result) {
  readout.classList.add('visible');

  const isNormal = result.prediction === 'No Failure';
  led.className = 'led ' + (isNormal ? 'safe' : 'alarm');
  statusText.innerHTML = isNormal
    ? 'Normal operation<span class="qualifier">No failure signature detected</span>'
    : result.prediction + '<span class="qualifier">Failure signature detected</span>';

  bars.innerHTML = '';
  const entries = Object.entries(result.scores || {});
  entries.forEach(([label, score], i) => {
    const row = document.createElement('div');
    row.className = 'bar-row' + (i === 0 ? ' top' : '');
    row.innerHTML =
      '<span>' + label + '</span>' +
      '<span class="bar-track"><span class="bar-fill" style="width:' + (score * 100).toFixed(1) + '%"></span></span>' +
      '<span class="bar-value">' + (score * 100).toFixed(1) + '%</span>';
    bars.appendChild(row);
  });
}

function renderError(message) {
  readout.classList.add('visible');
  led.className = 'led error';
  statusText.innerHTML = 'Diagnostic error<span class="qualifier">' + message + '</span>';
  bars.innerHTML = '';
}
</script>
</body>
</html>"""


@app.get("/")
def index():
    return INDEX_HTML


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        frame = build_input_frame(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    prediction = model.predict(frame)[0]
    scores = score_classes(model, frame)

    return jsonify({"prediction": prediction, "scores": scores})


if __name__ == "__main__":
    app.run(debug=True)
