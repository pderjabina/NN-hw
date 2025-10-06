// -------- Global state --------
let trainData = null;
let testData = null;
let pre = { trainX: null, trainY: null, testX: null, testIds: [] };
let model = null;
let valX = null, valY = null;
let valProbs = null;
let testProbs = null;

// -------- Schema (can be swapped for other datasets) --------
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

// -------- Robust CSV parser (handles quotes, commas, CRLF, BOM) --------
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
        out.push(cur);
        cur = '';
      } else {
        cur += ch;
      }
    }
    out.push(cur);
    return out;
  };

  const headers = splitLine(lines[0]).map(h => h.trim());
  const rows = [];
  for (let li = 1; li < lines.length; li++) {
    if (lines[li] === undefined || lines[li] === '') continue;
    const vals = splitLine(lines[li]).map(v => v === '' ? null : v);
    if (vals.length !== headers.length) continue; // skip malformed lines
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

async function loadBundled() {
  const status = document.getElementById('data-status');
  status.textContent = 'Loading bundled CSVs...';
  try {
    const trainText = document.getElementById('bundled-train').textContent;
    const testText  = document.getElementById('bundled-test').textContent;
    trainData = parseCSV(trainText);
    testData  = parseCSV(testText);
    status.textContent = `Bundled data loaded. Train: ${trainData.length}, Test: ${testData.length}`;
    document.getElementById('inspect-btn').disabled = false;
  } catch (e) {
    status.textContent = `Error loading bundled data: ${e.message}`;
    console.error(e);
  }
}

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

  stats.innerHTML = `<p>Shape: ${trainData.length} × ${cols}</p><p>Survival rate: ${survived}/${trainData.length} (${rate}%)</p>${miss}`;

  // Charts inside page
  renderSurvivalCharts();
  document.getElementById('preprocess-btn').disabled = false;
}

function renderBar(containerEl, data, opts) {
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
  const sexData = Object.entries(bySex).map(([k, v]) => ({ x: k, y: v.total ? (v.surv / v.total * 100) : 0 }));
  renderBar(document.getElementById('chart-sex'), sexData, { xLabel: 'Sex', yLabel: 'Survival Rate (%)' });

  // Pclass
  const byCls = {};
  trainData.forEach(r => {
    if (r.Pclass == null || r[TARGET] == null) return;
    byCls[r.Pclass] ??= { surv: 0, total: 0 };
    byCls[r.Pclass].total++;
    if (r[TARGET] === 1) byCls[r.Pclass].surv++;
  });
  const clsData = Object.entries(byCls).map(([k, v]) => ({ x: `Class ${k}`, y: v.total ? (v.surv / v.total * 100) : 0 }));
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
  model.compile({ optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy'] });

  const summary = document.getElementById('model-summary');
  let txt = '<h3>Model Summary</h3><ul>';
  model.layers.forEach((l,i)=>{ txt += `<li>Layer ${i+1}: ${l.getClassName()} — out: ${JSON.stringify(l.outputShape)}</li>`; });
  txt += `</ul><p>Total params: ${model.countParams()}</p>`;
  summary.innerHTML = txt;

  document.getElementById('train-btn').disabled = false;
}

// -------- Train --------
async function trainModel() {
  if (!model || !pre.trainX) { alert('Create model first.'); return; }
  const status = document.getElementById('training-status');
  status.textContent = 'Training...';

  // 80/20 split
  const n = pre.trainX.shape[0];
  const split = Math.floor(n * 0.8);
  const featDim = pre.trainX.shape[1];

  const trainX = pre.trainX.slice([0,0], [split, featDim]);
  const trainY = pre.trainY.slice([0], [split]);
  valX = pre.trainX.slice([split,0], [n - split, featDim]);
  valY = pre.trainY.slice([split], [n - split]);

  const vis = tfvis.show.fitCallbacks(
    { name: 'Training Performance', tab: 'Training' },
    ['loss','acc','val_loss','val_acc'],
    { callbacks: ['onEpochEnd'] }
  );
  const early = tf.callbacks.earlyStopping({ monitor:'val_loss', patience:5, restoreBestWeight:true });

  await model.fit(trainX, trainY, {
    epochs: 50, batchSize: 32, validationData: [valX, valY],
    callbacks: [vis, early, {
      onEpochEnd: (epoch, logs)=> {
        status.textContent = `Epoch ${epoch+1}/50 — loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
      }
    }]
  });

  status.innerHTML += '<p>Training completed.</p>';

  // cache validation probs (flattened)
  valProbs = model.predict(valX).dataSync();

  const slider = document.getElementById('threshold-slider');
  slider.disabled = true; // reset handler to avoid duplicate listeners
  slider.oninput = null;
  slider.disabled = false;
  slider.oninput = updateMetrics;
  updateMetrics();

  document.getElementById('predict-btn').disabled = false;
}

// -------- Metrics & ROC --------
function updateMetrics() {
  if (!valProbs || !valY) return;
  const thr = parseFloat(document.getElementById('threshold-slider').value);
  document.getElementById('threshold-value').textContent = thr.toFixed(2);

  const yTrue = Array.from(valY.dataSync());
  const probs  = Array.from(valProbs);

  let tp=0, tn=0, fp=0, fn=0;
  for (let i=0;i<probs.length;i++){
    const p = probs[i] >= thr ? 1 : 0;
    const t = yTrue[i];
    if (p===1 && t===1) tp++;
    else if (p===0 && t===0) tn++;
    else if (p===1 && t===0) fp++;
    else fn++;
  }

  // Confusion matrix
  document.getElementById('confusion-matrix').innerHTML =
    `<table>
      <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
      <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
      <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
     </table>`;

  const precision = tp / (tp + fp) || 0;
  const recall    = tp / (tp + fn) || 0;
  const f1        = 2 * (precision * recall) / (precision + recall) || 0;
  const acc       = (tp + tn) / (tp + tn + fp + fn) || 0;

  document.getElementById('performance-metrics').innerHTML =
    `<p>Accuracy: ${(acc*100).toFixed(2)}%</p>
     <p>Precision: ${precision.toFixed(4)}</p>
     <p>Recall: ${recall.toFixed(4)}</p>
     <p>F1 Score: ${f1.toFixed(4)}</p>`;

  plotROC(yTrue, probs);
}

function plotROC(yTrue, probs) {
  const thresholds = Array.from({length:101}, (_,i)=> i/100);
  const pts = thresholds.map(th=>{
    let tp=0, fp=0, tn=0, fn=0;
    for (let i=0;i<probs.length;i++){
      const p = probs[i] >= th ? 1 : 0;
      const t = yTrue[i];
      if (t===1) { if (p===1) tp++; else fn++; }
      else { if (p===1) fp++; else tn++; }
    }
    const tpr = tp / (tp + fn) || 0;
    const fpr = fp / (fp + tn) || 0;
    return {x: fpr, y: tpr};
  }).sort((a,b)=>a.x-b.x);

  // AUC trapezoid
  let auc = 0;
  for (let i=1;i<pts.length;i++){
    const dx = pts[i].x - pts[i-1].x;
    const mY = (pts[i].y + pts[i-1].y)/2;
    auc += dx * mY;
  }

  tfvis.render.linechart(
    document.getElementById('roc-chart'),
    { values: pts, series: ['ROC'] },
    { xLabel:'False Positive Rate', yLabel:'True Positive Rate', width:480, height:320 }
  );

  const pm = document.getElementById('performance-metrics');
  const aucP = pm.querySelector('.auc-line');
  const line = `<p class="auc-line">AUC: ${auc.toFixed(4)}</p>`;
  if (aucP) aucP.outerHTML = line; else pm.innerHTML += line;
}

// -------- Predict & Export --------
async function predict() {
  if (!model || !pre.testX) { alert('Train model first.'); return; }
  const out = document.getElementById('prediction-output');
  out.textContent = 'Predicting...';

  const probs = model.predict(pre.testX).dataSync();
  testProbs = probs;

  const rows = pre.testIds.map((id,i)=>({
    PassengerId: id,
    Survived: (probs[i] >= 0.5) ? 1 : 0,
    Probability: probs[i]
  }));

  const table = document.createElement('table');
  const head = document.createElement('tr');
  ['PassengerId','Survived','Probability'].forEach(h=>{ const th=document.createElement('th'); th.textContent=h; head.appendChild(th); });
  table.appendChild(head);
  rows.slice(0,10).forEach(r=>{
    const tr = document.createElement('tr');
    ['PassengerId','Survived','Probability'].forEach(k=>{
      const td=document.createElement('td');
      td.textContent = (k==='Probability') ? r[k].toFixed(4) : r[k];
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });

  out.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
  out.appendChild(table);
  out.innerHTML += `<p>Total predictions: ${rows.length}</p>`;

  document.getElementById('export-btn').disabled = false;
}

async function exportResults() {
  if (!testProbs || !pre.testIds.length) { alert('Make predictions first.'); return; }
  const status = document.getElementById('export-status');
  status.textContent = 'Exporting...';

  const sub = ['PassengerId,Survived']
    .concat(pre.testIds.map((id,i)=>`${id},${testProbs[i] >= 0.5 ? 1 : 0}`))
    .join('\n');

  const probsCsv = ['PassengerId,Probability']
    .concat(pre.testIds.map((id,i)=>`${id},${testProbs[i].toFixed(6)}`))
    .join('\n');

  const dl = (name, text) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], {type:'text/csv'}));
    a.download = name;
    a.click();
  };
  dl('submission.csv', sub);
  dl('probabilities.csv', probsCsv);

  await model.save('downloads://titanic-tfjs-model');
  status.innerHTML = `<p>Export completed.</p>
                      <p>Saved: submission.csv, probabilities.csv</p>
                      <p>Model downloaded as titanic-tfjs-model*</p>`;
}
