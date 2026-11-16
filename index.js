'use strict';
// Runs the two ONNX models with onnxruntime-web, following the class pattern:
// Float32Array -> ort.Tensor('float32', data, [rows, n]) -> session.run({ input1 }) -> results.output1.data

const REGIONS = ['northeast', 'northwest', 'southeast', 'southwest'];
const DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                           'DiabetesPedigreeFunction', 'Age'];
const FILLED_IF_ZERO = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'];

// ---- generated from the notebooks' results JSON ----
const DIABETES_THRESHOLD = 0.48;   // chosen on out-of-fold predictions for best F1
const INSURANCE_FACTS = "Model: ShallowTanh (PyTorch), chosen by 5-fold cross-validation (mean R² 0.81). On 268 held-out people: R² = 0.89, mean absolute error = $2,398.";
const DIABETES_FACTS = "Model: F1plusF2 (PyTorch), decision threshold 0.48 chosen by cross-validation. On 154 held-out patients: F1 = 0.66, recall = 0.74, precision = 0.60, accuracy = 0.73, ROC-AUC = 0.81.";
// held-out test rows with the prediction ONNX Runtime (Python) gave
const INSURANCE_SAMPLES = [
  {
    "inputs": {
      "age": 49,
      "bmi": 22.515,
      "children": 0,
      "sex": "male",
      "smoker": "no",
      "region": "northeast"
    },
    "actual": 8688.86,
    "onnx_pred": 9622.04
  },
  {
    "inputs": {
      "age": 31,
      "bmi": 34.39,
      "children": 3,
      "sex": "male",
      "smoker": "yes",
      "region": "northwest"
    },
    "actual": 38746.36,
    "onnx_pred": 33709.15
  },
  {
    "inputs": {
      "age": 29,
      "bmi": 25.6,
      "children": 4,
      "sex": "female",
      "smoker": "no",
      "region": "southwest"
    },
    "actual": 5708.87,
    "onnx_pred": 5944.05
  }
];
const DIABETES_SAMPLES = [
  {
    "inputs": {
      "Pregnancies": 7,
      "Glucose": 114,
      "BloodPressure": 64,
      "SkinThickness": 0,
      "Insulin": 0,
      "BMI": 27.4,
      "DiabetesPedigreeFunction": 0.732,
      "Age": 34
    },
    "actual": 1,
    "onnx_prob": 0.395201
  },
  {
    "inputs": {
      "Pregnancies": 7,
      "Glucose": 159,
      "BloodPressure": 64,
      "SkinThickness": 0,
      "Insulin": 0,
      "BMI": 27.4,
      "DiabetesPedigreeFunction": 0.294,
      "Age": 40
    },
    "actual": 0,
    "onnx_prob": 0.733094
  },
  {
    "inputs": {
      "Pregnancies": 8,
      "Glucose": 105,
      "BloodPressure": 100,
      "SkinThickness": 36,
      "Insulin": 0,
      "BMI": 43.3,
      "DiabetesPedigreeFunction": 0.239,
      "Age": 45
    },
    "actual": 1,
    "onnx_prob": 0.664806
  }
];


// 9 inputs, same order as FEATURES in Part A: age, bmi, children, sex_male, smoker_yes, region one-hot (4)
function encodeInsurance(p) {
  const x = new Float32Array(9);
  x[0] = Number(p.age); x[1] = Number(p.bmi); x[2] = Number(p.children);
  x[3] = p.sex === 'male' ? 1 : 0;
  x[4] = p.smoker === 'yes' ? 1 : 0;
  const r = REGIONS.indexOf(p.region);
  if (r < 0) throw new Error('Unknown region: ' + p.region);
  x[5 + r] = 1;
  return x;
}

// 8 inputs in DIABETES_FEATURES order; zeros are passed through and imputed inside the model
function encodeDiabetes(p) {
  return Float32Array.from(DIABETES_FEATURES, (k) => Number(p[k]));
}

const MODELS = {
  insurance: { url: './insurance_charges.onnx', width: 9, encode: encodeInsurance },
  diabetes:  { url: './diabetes_risk.onnx',     width: 8, encode: encodeDiabetes },
};

const sessions = {};
function session(key) {
  if (!sessions[key]) sessions[key] = ort.InferenceSession.create(MODELS[key].url);
  return sessions[key];
}

// run one or more people/patients in a single batch; returns one number per row
async function runModel(key, people) {
  const m = MODELS[key];
  const data = new Float32Array(people.length * m.width);
  people.forEach((p, i) => data.set(m.encode(p), i * m.width));
  const s = await session(key);
  const out = await s.run({ input1: new ort.Tensor('float32', data, [people.length, m.width]) });
  return Array.from(out.output1.data);
}

// ---------------- page ----------------
const $ = (id) => document.getElementById(id);
const money = (v) => '$' + Math.round(v).toLocaleString('en-US');
const radio = (name) => document.querySelector(`input[name="${name}"]:checked`).value;
const setRadio = (name, v) => { document.querySelector(`input[name="${name}"][value="${v}"]`).checked = true; };

function numbers(ids, prefix) {
  const out = {}, bad = [];
  for (const id of ids) {
    const el = $(prefix + id);
    const v = parseFloat(el.value);
    const ok = el.value.trim() !== '' && Number.isFinite(v) && v >= 0;
    el.setAttribute('aria-invalid', ok ? 'false' : 'true');
    if (!ok) bad.push(el.closest('label').firstChild.textContent.trim().toLowerCase());
    out[id] = v;
  }
  return { out, bad };
}

function fail(key, msg) {
  const box = $('out-' + key);
  box.className = 'result';
  box.innerHTML = `<p class="label">Check the form</p><p class="small" style="color:var(--red)">${msg}</p>`;
}

let pendingCheck = { insurance: null, diabetes: null };

async function onInsurance(e) {
  if (e) e.preventDefault();
  const { out, bad } = numbers(['age', 'bmi', 'children'], 'i-');
  if (bad.length) return fail('insurance', 'Enter a valid number for ' + bad.join(', ') + '.');
  const person = { ...out, sex: radio('i-sex'), smoker: radio('i-smoker'), region: $('i-region').value };
  const other = { ...person, smoker: person.smoker === 'yes' ? 'no' : 'yes' };
  $('run-insurance').disabled = true;
  try {
    const [usd, otherUsd] = await runModel('insurance', [person, other]);
    const smoker = person.smoker === 'yes';
    const check = pendingCheck.insurance;
    const box = $('out-insurance');
    box.className = 'result on ins';
    box.innerHTML = `
      <p class="label">Estimated yearly charges</p>
      <p class="amount">${money(usd)}</p>
      <p class="compare">As a ${smoker ? 'non-smoker' : 'smoker'}, the same person would be charged about <strong>${money(otherUsd)}</strong>
        (${smoker ? (usd / otherUsd).toFixed(1) + '× less' : (otherUsd / usd).toFixed(1) + '× more'}).</p>
      ${(person.age < 18 || person.age > 64) ? '<p class="small">This age is outside the 18–64 range the model was trained on.</p>' : ''}
      ${check ? `<p class="check">Test-set person: actually charged ${money(check.actual)}. The notebook’s ONNX Runtime run predicted ${money(check.onnx_pred)}.</p>` : ''}`;
  } catch (err) { console.error(err); fail('insurance', 'The model did not run: ' + err.message); }
  finally { $('run-insurance').disabled = false; pendingCheck.insurance = null; }
}

async function onDiabetes(e) {
  if (e) e.preventDefault();
  const { out, bad } = numbers(DIABETES_FEATURES, 'd-');
  if (bad.length) return fail('diabetes', 'Enter a valid number for ' + bad.join(', ') + '.');
  $('run-diabetes').disabled = true;
  try {
    const [p] = await runModel('diabetes', [out]);
    const high = p >= DIABETES_THRESHOLD;
    const filled = FILLED_IF_ZERO.filter((k) => out[k] === 0);
    const check = pendingCheck.diabetes;
    const box = $('out-diabetes');
    box.className = 'result on ' + (high ? 'high' : 'low');
    const pct = (p * 100).toFixed(0), t = (DIABETES_THRESHOLD * 100).toFixed(0);
    box.innerHTML = `
      <p class="label">Probability of diabetes</p>
      <p class="amount">${pct}%</p>
      <p class="compare"><strong>${high ? 'Flagged as likely diabetic' : 'Not flagged'}</strong>. The model flags patients at ${DIABETES_THRESHOLD.toFixed(2)} or above, the threshold that gave the best cross-validated F1.</p>
      <div class="meter" style="color:${high ? 'var(--red)' : 'var(--green)'}" aria-hidden="true">
        <i style="width:${pct}%"></i><b style="left:calc(${t}% - 1px)"></b><em style="left:${t}%">threshold</em>
      </div>
      <div class="ends"><span>0%</span><span>100%</span></div>
      ${filled.length ? `<p class="small">Filled in from training medians: ${filled.join(', ')}.</p>` : ''}
      ${check ? `<p class="check">Test-set patient: ${check.actual ? 'diabetic' : 'not diabetic'}. The notebook’s ONNX Runtime run gave ${(check.onnx_prob * 100).toFixed(1)}%.</p>` : ''}`;
  } catch (err) { console.error(err); fail('diabetes', 'The model did not run: ' + err.message); }
  finally { $('run-diabetes').disabled = false; pendingCheck.diabetes = null; }
}

function showTab(key, focus) {
  for (const k of ['insurance', 'diabetes']) {
    const on = k === key;
    $('tab-' + k).setAttribute('aria-selected', on ? 'true' : 'false');
    $('tab-' + k).tabIndex = on ? 0 : -1;
    $('panel-' + k).hidden = !on;
  }
  if (focus) $('tab-' + key).focus();
  if (location.hash !== '#' + key) history.replaceState(null, '', '#' + key);
}

function addSampleButtons(key, list, load) {
  list.forEach((s, i) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'chip'; b.textContent = String(i + 1);
    b.setAttribute('aria-label', 'Load test-set example ' + (i + 1));
    b.addEventListener('click', () => { pendingCheck[key] = s; load(s.inputs); });
    $('samples-' + key).append(b, ' ');
  });
}

function repoLinks() {
  // On <user>.github.io/<repo>/ point the notebook links at GitHub's rendered view
  const host = location.hostname, parts = location.pathname.split('/').filter(Boolean);
  if (!host.endsWith('.github.io') || !parts.length) return;
  const base = `https://github.com/${host.split('.')[0]}/${parts[0]}/blob/main/`;
  $('link-a').href = base + 'PartA_Regression_InsuranceCharges.ipynb';
  $('link-b').href = base + 'PartB_Classification_Diabetes.ipynb';
}

async function init() {
  if (init.started) return;      // run once even if DOMContentLoaded fires twice
  init.started = true;
  ort.env.wasm.numThreads = 1;
  $('form-insurance').addEventListener('submit', onInsurance);
  $('form-diabetes').addEventListener('submit', onDiabetes);
  for (const k of ['insurance', 'diabetes']) $('tab-' + k).addEventListener('click', () => showTab(k));
  document.querySelector('[role="tablist"]').addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      showTab($('tab-insurance').getAttribute('aria-selected') === 'true' ? 'diabetes' : 'insurance', true);
    }
  });
  window.addEventListener('hashchange', () => showTab(location.hash === '#diabetes' ? 'diabetes' : 'insurance'));
  if (location.hash === '#diabetes') showTab('diabetes');

  addSampleButtons('insurance', INSURANCE_SAMPLES, (p) => {
    $('i-age').value = p.age; $('i-bmi').value = p.bmi; $('i-children').value = p.children;
    $('i-region').value = p.region; setRadio('i-sex', p.sex); setRadio('i-smoker', p.smoker);
    onInsurance();
  });
  addSampleButtons('diabetes', DIABETES_SAMPLES, (p) => {
    for (const k of DIABETES_FEATURES) $('d-' + k).value = p[k];
    onDiabetes();
  });
  $('facts-insurance').textContent = INSURANCE_FACTS;
  $('facts-diabetes').textContent = DIABETES_FACTS;
  repoLinks();

  try {
    await Promise.all([session('insurance'), session('diabetes')]);
    $('status').textContent = 'Both models loaded. Pick a tab to begin.';
  } catch (err) {
    console.error(err);
    $('status').className = 'bad';
    $('status').textContent = 'The .onnx files did not load. Serve this folder over http(s), for example from GitHub Pages ' +
      'or with “python -m http.server”; browsers block model files opened straight from disk.';
  }
}

if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', init);
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { encodeInsurance, encodeDiabetes, MODELS,
                     get DIABETES_THRESHOLD() { return DIABETES_THRESHOLD; },
                     INSURANCE_SAMPLES, DIABETES_SAMPLES };
}
