// -------- Global state --------
let trainData = null;
let testData = null;
let pre = { trainX: null, trainY: null, testX: null, testIds: [] };
let model = null;
let valX = null, valY = null;
let valProbs = null;
let testProbs = null;

// -------- Schema --------
const TARGET = 'Survived';
const ID = 'PassengerId';

// -------- DOM wiring --------
document.getElementById('load-bundled-btn').onclick = loadBundled;
document.getElementById('load-data-btn').onclick = loadUploaded;
document.getElementById('inspect-btn').onclick = inspectData;
document.getElementById('preprocess-btn').onclick = preprocessData;
document.getElementById('create-model-btn').onclick = createModel;
document.getElementById('train-btn').onclick = trainModel;
document.getElementById('predict-btn').onclick = predict;
document.getElementById('export-btn').onclick = exportResults;

// -------- Robust CSV parser (quotes, commas, CRLF, BOM) --------
function parseCSV(text) {
  text = text.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').trim();
  const lines = text.split('\n');

  const splitLine = (line) => {
    const out = [];
    let cur = '', inQ = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQ && line[i + 1] === '"') { cur += '"'; i++; }
        else { inQ = !inQ; }
      } else if (ch === ',' && !inQ) {
        out.push(cur); cur = '';
      } else { cur += ch; }
    }
    out.push(cur);
    return out;
  };

  const headers = splitLine(lines[0]).map(h => h.trim());
  const rows = [];
  for (let li = 1; li < lines.length; li++) {
    if (lines[li] === undefined || lines[li] === '') continue;
    const vals = splitLine(lines[li]).map(v => v === '' ? null : v);
    if (vals.length !== headers.length) continue; // skip malformed
    const obj = {};
    headers.forEach((h, i) => {
      let v = vals[i] ?? null;
      if (v !== null && !isNaN(v) && v.trim() !== '') v = parseFloat(v);
      obj[h] = v;
    });
    rows.push(obj);
  }
  return rows;
}

// -------- Loaders --------
function readFile(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = e => res(e.target.result);
    r.onerror = () => rej(new Error('Failed to read file'));
    r.readAsText(file);
  });
}

// Option A: fetch train.csv & test.csv from repo root (if present)
async function loadBundled() {
  const status = document.getElementById('data-status');
  status.textContent = 'Loading bundled CSVs...';
  try {
    const [trainText, testText] = await Promise.all([
      fetch('train.csv').then(r => { if(!r.ok) throw new Error('train.csv not found'); return r.text(); }),
      fetch('test.csv').then(r  => { if(!r.ok) throw new Error('test.csv not found');  return r.text(); })
    ]);
    trainData = parseCSV(trainText);
    testData  = parseCSV(testText);
    status.textContent = `Bundled data loaded. Train: ${trainData.length}, Test: ${testData.length}`;
    document.getElementById('inspect-btn').disabled = false;
  } catch (e) {
    status.textContent = `Bundled load failed: ${e.message}. Use the upload inputs below.`;
    console.error(e);
  }
}

// Option B: user uploads both files
async function loadUploaded() {
  const trainFile = document.getElementById('train-file').files[0];
  const testFile  = document.getElementById('test-file').files[0];
  const status = document.getElementById('data-status');

  if (!trainFile || !testFile) { alert('Please upload both training and test CSV files.'); return; }
  status.textContent = 'Loading uploaded CSVs...';

  try {
    const [trainText, testText] = await Promise.all([readFile(trainFile), readFile(testFile)]);
    trainData = parseCSV(trainText);
    testData  = parseCSV(testText);
    status.textContent = `Uploaded data loaded. Train: ${trainData.length}, Test: ${testData.length}`;
    document.getElementById('inspect-btn').disabled = false;
  } catch (e) {
    status.textContent = `Error loading files: ${e.message}`;
    console.error(e);
  }
}

// -------- Inspection --------
function tableFromObjects(arr) {
  const table = document.createElement('table');
  const thead = document.createElement('tr');
  Object.keys(arr[0]).forEach(k => { const th = document.createElement('th'); th.textContent = k; thead.appendChild(th); });
  table.appendChild(thead);
  arr.forEach(row => {
    const tr = document.createElement('tr');
    Object.keys(arr[0]).forEach(k => {
      const td = document.createElement('td');
      const v = row[k];
      td.textContent = (v === null || v === undefined) ? 'NULL' : v;
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  return table;
}

function inspectData() {
  if (!trainData?.length) { alert('Please load data first.'); return; }

  const preview = document.getElementById('data-preview');
  preview.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
  preview.appendChild(tableFromObjects(trainData.slice(0, 10)));

  const stats = document.getElementById('data-stats');
  const cols = Object.keys(trainData[0]).length;
  const survived = trainData.filter(r => r[TARGET] === 1).length;
  const rate = (survived / trainData.length * 100).toFixed(2);

  let miss = '<h4>Missing Values (%):</h4><ul>';
  Object.keys(trainData[0]).forEach(f => {
    const m = trainData.reduce((c, r) => c + ((r[f] === null || r[f] === undefined) ? 1 : 0), 0);
    miss += `<li>${f}: ${(m / trainData.length * 100).toFixed(2)}%</li>`;
  });
  miss += '</ul>';

  stats.innerHTML = `<p>Shape: ${trainData.length} × ${cols}</p>
                     <p>Survival rate: ${survived}/${trainData.length} (${rate}%)</p>
                     ${miss}
                     <p style="color:#6b7280">Next: click <b>Preprocess Data</b> → <b>Create Model</b> → <b>Train Model</b>.</p>`;

  renderSurvivalCharts();
  document.getElementById('preprocess-btn').disabled = false;
}

function renderBar(containerEl, data, opts) {
  // tfjs-vis expects objects with "index" and "value" keys for barchart
  tfvis.render.barchart(containerEl, data, opts);
}

function renderSurvivalCharts() {
  // Sex
  const bySex = {};
  trainData.forEach(r => {
    if (r.Sex == null || r[TARGET] == null) return;
    bySex[r.Sex] ??= { surv: 0, total: 0 };
    bySex[r.Sex].total++;
    if (r[TARGET] === 1) bySex[r.Sex].surv++;
  });
  const sexData = Object.entries(bySex).map(([k, v]) => ({ index: String(k), value: v.total ? (v.surv / v.total * 100) : 0 }));
  renderBar(document.getElementById('chart-sex'), sexData, { xLabel: 'Sex', yLabel: 'Survival Rate (%)' });

  // Pclass
  const byCls = {};
  trainData.forEach(r => {
    if (r.Pclass == null || r[TARGET] == null) return;
    byCls[r.Pclass] ??= { surv: 0, total: 0 };
    byCls[r.Pclass].total++;
    if (r[TARGET] === 1) byCls[r.Pclass].surv++;
  });
  const clsData = Object.entries(byCls).map(([k, v]) => ({ index: `Class ${k}`, value: v.total ? (v.surv / v.total * 100) : 0 }));
  renderBar(document.getElementById('chart-pclass'), clsData, { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' });
}

// -------- Preprocessing --------
function median(arr) {
  const v = arr.filter(x => x != null).slice().sort((a,b)=>a-b);
  if (!v.length) return 0;
  const h = Math.floor(v.length/2);
  return (v.length%2)? v[h] : (v[h-1]+v[h])/2;
}
function mode(arr) {
  const v = arr.filter(x => x != null);
  if (!v.length) return null;
  const f = {}; let m = null, c = 0;
  v.forEach(x => { f[x]=(f[x]||0)+1; if (f[x]>c){c=f[x]; m=x;} });
  return m;
}
function std(arr) {
  const v = arr.filter(x => x != null);
  if (!v.length) return 1;
  const mu = v.reduce((s,x)=>s+x,0)/v.length;
  const va = v.reduce((s,x)=>s+(x-mu)**2,0)/v.length;
  return Math.sqrt(va)||1;
}
function oneHot(value, cats) {
  const a = new Array(cats.length).fill(0);
  const i = cats.indexOf(value);
  if (i >= 0) a[i] = 1;
  return a;
}

function preprocessData() {
  if (!trainData || !testData) { alert('Please load data first.'); return; }
  const out = document.getElementById('preprocessing-output');
  out.textContent = 'Preprocessing...';

  // stats from train
  const ageMed = median(trainData.map(r => r.Age));
  const fareMed = median(trainData.map(r => r.Fare));
  const ageStd = std(trainData.map(r => r.Age));
  const fareStd = std(trainData.map(r => r.Fare));
  const embarkedMode = mode(trainData.map(r => r.Embarked));

  const addFamily = document.getElementById('add-family-features').checked;

  const buildX = (row) => {
    const age = (row.Age != null) ? row.Age : ageMed;
    const fare = (row.Fare != null) ? row.Fare : fareMed;
    const embarked = (row.Embarked != null) ? row.Embarked : embarkedMode;

    const zAge = (age - ageMed) / ageStd;
    const zFare = (fare - fareMed) / fareStd;

    let feat = [ zAge, zFare, row.SibSp ?? 0, row.Parch ?? 0 ];
    feat = feat
      .concat(oneHot(row.Pclass, [1,2,3]))
      .concat(oneHot(row.Sex, ['male','female']))
      .concat(oneHot(embarked, ['C','Q','S']));

    if (addFamily) {
      const familySize = (row.SibSp ?? 0) + (row.Parch ?? 0) + 1;
      const isAlone = familySize === 1 ? 1 : 0;
      feat.push(familySize, isAlone);
    }
    return feat;
  };

  const X = trainData.map(buildX);
  const y = trainData.map(r => r[TARGET]);

  pre.trainX = tf.tensor2d(X);
  pre.trainY = tf.tensor1d(y, 'float32');

  pre.testX = tf.tensor2d(testData.map(buildX));
  pre.testIds = testData.map(r => r[ID]);

  out.innerHTML =
    `<p>Done.</p>
     <p>Train X shape: ${pre.trainX.shape}</p>
     <p>Train y shape: ${pre.trainY.shape}</p>
     <p>Test  X shape: ${pre.testX.shape}</p>`;

  document.getElementById('create-model-btn').disabled = false;
}

// -------- Model --------
function createModel() {
  if (!pre.trainX) { alert('Preprocess first.'); return; }
  model = tf.sequential();
  model.add(tf.layers.dense({ units:16, activation:'relu', inputShape:[pre.trainX.shape[1]] }));
  model.add(tf.layers.dense({ units:1, activation:'sigmoid' }));
  model.compile({ optim
