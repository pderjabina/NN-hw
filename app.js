// === Titanic TensorFlow.js Classifier (final) ===
let trainData = null, testData = null;
let pre = {trainX:null, trainY:null, testX:null, testIds:[]};
let model = null, valX=null, valY=null, valProbs=null, testProbs=null;

const TARGET = 'Survived';
const ID = 'PassengerId';

// buttons
document.getElementById('load-bundled-btn').onclick = loadBundled;
document.getElementById('load-data-btn').onclick = loadUploaded;
document.getElementById('inspect-btn').onclick = inspectData;
document.getElementById('preprocess-btn').onclick = preprocessData;
document.getElementById('create-model-btn').onclick = createModel;
document.getElementById('train-btn').onclick = trainModel;
document.getElementById('predict-btn').onclick = predict;
document.getElementById('export-btn').onclick = exportResults;

/* ---------------- CSV PARSER (устойчив к кавычкам/запятым/CRLF/BOM) ---------------- */
function parseCSV(text){
  text = text.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').trim();
  const lines = text.split('\n');
  const split = (line) => {
    const out = []; let cur = '', inQ = false;
    for (let i=0;i<line.length;i++){
      const ch=line[i];
      if (ch === '"') { if (inQ && line[i+1] === '"'){ cur+='"'; i++; } else inQ=!inQ; }
      else if (ch === ',' && !inQ) { out.push(cur); cur=''; }
      else cur+=ch;
    }
    out.push(cur); return out;
  };
  const headers = split(lines[0]).map(h=>h.trim());
  const rows = [];
  for (let i=1;i<lines.length;i++){
    if (!lines[i]) continue;
    const vals = split(lines[i]).map(v => v==='' ? null : v);
    const obj = {};
    headers.forEach((h, j) => {
      let v = vals[j] ?? null;
      if (v !== null && !isNaN(v) && v.trim()!=='') v = parseFloat(v);
      obj[h] = v;
    });
    rows.push(obj);
  }
  return rows;
}

/* ---------------- LOADERS ---------------- */
// A) взять train.csv/test.csv из корня репозитория (то, что ты добавила)
async function loadBundled(){
  const s = document.getElementById('data-status');
  try{
    s.textContent = 'Loading bundled CSVs...';
    const [tr, te] = await Promise.all([ fetch('train.csv'), fetch('test.csv') ]);
    if (!tr.ok || !te.ok) throw new Error('train.csv/test.csv not found in repo root');
    trainData = parseCSV(await tr.text());
    testData  = parseCSV(await te.text());
    s.textContent = `Bundled data loaded. Train: ${trainData.length}, Test: ${testData.length}`;
    document.getElementById('inspect-btn').disabled = false;
  }catch(e){
    s.textContent = 'Bundled load failed: '+e.message;
    console.error(e);
  }
}

// B) загрузка через inputs (если нужно)
function readFile(f){return new Promise((res,rej)=>{const r=new FileReader();r.onload=e=>res(e.target.result);r.onerror=()=>rej();r.readAsText(f);});}
async function loadUploaded(){
  const tr = document.getElementById('train-file').files[0];
  const te = document.getElementById('test-file').files[0];
  const s  = document.getElementById('data-status');
  if(!tr||!te){alert('Upload both files');return;}
  s.textContent='Loading uploaded CSVs...';
  try{
    const [a,b] = await Promise.all([readFile(tr), readFile(te)]);
    trainData=parseCSV(a); testData=parseCSV(b);
    s.textContent=`Uploaded data loaded. Train: ${trainData.length}, Test: ${testData.length}`;
    document.getElementById('inspect-btn').disabled=false;
  }catch(e){ s.textContent='Error loading files'; }
}

/* ---------------- INSPECTION ---------------- */
function tableFromObjects(arr){
  const tbl=document.createElement('table'), head=document.createElement('tr');
  Object.keys(arr[0]).forEach(k=>{const th=document.createElement('th'); th.textContent=k; head.appendChild(th);});
  tbl.appendChild(head);
  arr.forEach(r=>{
    const tr=document.createElement('tr');
    Object.keys(arr[0]).forEach(k=>{const td=document.createElement('td'); td.textContent = (r[k]==null?'NULL':r[k]); tr.appendChild(td);});
    tbl.appendChild(tr);
  });
  return tbl;
}

function inspectData(){
  if(!trainData?.length){alert('Load data first');return;}
  const prev = document.getElementById('data-preview');
  prev.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
  prev.appendChild(tableFromObjects(trainData.slice(0,10)));

  // графики выживаемости по полу и классу
  const bySex={}, byCls={};
  trainData.forEach(r=>{
    if(r.Sex!=null){ bySex[r.Sex] ??= {s:0,t:0}; bySex[r.Sex].t++; if(r[TARGET]===1) bySex[r.Sex].s++; }
    if(r.Pclass!=null){ byCls[r.Pclass] ??= {s:0,t:0}; byCls[r.Pclass].t++; if(r[TARGET]===1) byCls[r.Pclass].s++; }
  });
  const sex = Object.entries(bySex).map(([k,v])=>({ index:String(k),   value: v.t ? v.s/v.t*100 : 0 }));
  const cls = Object.entries(byCls).map(([k,v])=>({ index:`Class ${k}`, value: v.t ? v.s/v.t*100 : 0 }));
  tfvis.render.barchart(document.getElementById('chart-sex'),   sex, {xLabel:'Sex', yLabel:'Survival Rate (%)'});
  tfvis.render.barchart(document.getElementById('chart-pclass'), cls, {xLabel:'Passenger Class', yLabel:'Survival Rate (%)'});

  document.getElementById('preprocess-btn').disabled = false;
}

/* ---------------- PREPROCESS ---------------- */
function median(a){a=a.filter(x=>x!=null).sort((x,y)=>x-y); if(!a.length) return 0; const h=Math.floor(a.length/2); return a.length%2?a[h]:(a[h-1]+a[h])/2;}
function std(a){a=a.filter(x=>x!=null); if(!a.length) return 1; const m=a.reduce((s,x)=>s+x,0)/a.length; const v=a.reduce((s,x)=>s+(x-m)**2,0)/a.length; return Math.sqrt(v)||1;}
function mode(a){const f={}; a.filter(x=>x!=null).forEach(x=>f[x]=(f[x]||0)+1); return Object.entries(f).sort((a,b)=>b[1]-a[1])[0]?.[0] ?? null;}
function oneHot(v,c){const arr=new Array(c.length).fill(0); const i=c.indexOf(v); if(i>=0) arr[i]=1; return arr;}

function preprocessData(){
  const out=document.getElementById('preprocessing-output'); out.textContent='Preprocessing...';
  const ageM=median(trainData.map(r=>r.Age)),  fareM=median(trainData.map(r=>r.Fare));
  const ageS=std(trainData.map(r=>r.Age)),     fareS=std(trainData.map(r=>r.Fare));
  const embM=mode(trainData.map(r=>r.Embarked));
  const add=document.getElementById('add-family-features').checked;

  const build=(r)=>{
    const age = ((r.Age!=null?r.Age:ageM) - ageM)/ageS;
    const fare= ((r.Fare!=null?r.Fare:fareM) - fareM)/fareS;
    let f=[age,fare, r.SibSp??0, r.Parch??0];
    f=f
      .concat(oneHot(r.Pclass,[1,2,3]))
      .concat(oneHot(r.Sex,['male','female']))
      .concat(oneHot(r.Embarked??embM,['C','Q','S']));
    if(add){ const fs=(r.SibSp??0)+(r.Parch??0)+1; f.push(fs, fs===1?1:0); }
    return f;
  };

  pre.trainX=tf.tensor2d(trainData.map(build));
  pre.trainY=tf.tensor1d(trainData.map(r=>r[TARGET]),'float32');
  pre.testX =tf.tensor2d(testData.map(build));
  pre.testIds=testData.map(r=>r[ID]);

  out.innerHTML = `<p>Done.</p>
                   <p>Train X shape: ${pre.trainX.shape}</p>
                   <p>Train y shape: ${pre.trainY.shape}</p>
                   <p>Test  X shape: ${pre.testX.shape}</p>`;
  document.getElementById('create-model-btn').disabled = false;
}

/* ---------------- MODEL / TRAIN ---------------- */
function createModel(){
  model=tf.sequential();
  model.add(tf.layers.dense({units:16,activation:'relu',inputShape:[pre.trainX.shape[1]]}));
  model.add(tf.layers.dense({units:1,activation:'sigmoid'}));
  model.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});
  document.getElementById('train-btn').disabled = false;
}

async function trainModel(){
  const n=pre.trainX.shape[0], split=Math.floor(n*0.8), d=pre.trainX.shape[1];
  const trX=pre.trainX.slice([0,0],[split,d]), trY=pre.trainY.slice([0],[split]);
  valX=pre.trainX.slice([split,0],[n-split,d]); valY=pre.trainY.slice([split],[n-split]);

  const vis=tfvis.show.fitCallbacks({name:'Training'},['loss','acc','val_loss','val_acc'],{callbacks:['onEpochEnd']});
  const early=tf.callbacks.earlyStopping({monitor:'val_loss',patience:5,restoreBestWeight:true});
  await model.fit(trX,trY,{epochs:50,batchSize:32,validationData:[valX,valY],callbacks:[vis,early]});

  valProbs = model.predict(valX).dataSync();
  const slider=document.getElementById('threshold-slider');
  slider.disabled=false; slider.oninput=updateMetrics;
  updateMetrics();
  document.getElementById('predict-btn').disabled=false;
}

/* ---------------- METRICS ---------------- */
function updateMetrics(){
  if(!valProbs||!valY) return;
  const thr=parseFloat(document.getElementById('threshold-slider').value);
  document.getElementById('threshold-value').textContent=thr.toFixed(2);

  const y=Array.from(valY.dataSync()), p=Array.from(valProbs);
  let tp=0,tn=0,fp=0,fn=0;
  for(let i=0;i<p.length;i++){
    const pr=p[i]>=thr?1:0, t=y[i];
    if(pr&&t) tp++; else if(!pr&&!t) tn++; else if(pr&&!t) fp++; else fn++;
  }

  document.getElementById('confusion-matrix').innerHTML =
    `<table>
      <tr><th></th><th>Pred +</th><th>Pred -</th></tr>
      <tr><th>True +</th><td>${tp}</td><td>${fn}</td></tr>
      <tr><th>True -</th><td>${fp}</td><td>${tn}</td></tr>
     </table>`;

  const prec=tp/(tp+fp)||0, rec=tp/(tp+fn)||0, f1=2*(prec*rec)/(prec+rec)||0, acc=(tp+tn)/(tp+tn+fp+fn)||0;
  document.getElementById('performance-metrics').innerHTML =
    `<p>Accuracy: ${(acc*100).toFixed(2)}%</p>
     <p>Precision: ${prec.toFixed(3)}</p>
     <p>Recall: ${rec.toFixed(3)}</p>
     <p>F1: ${f1.toFixed(3)}</p>`;

  plotROC(y,p);
}

function plotROC(yTrue, probs){
  const thresholds = Array.from({length:101},(_,i)=>i/100);
  const pts = thresholds.map(t=>{
    let tp=0,fp=0,tn=0,fn=0;
    for(let i=0;i<probs.length;i++){
      const p = probs[i]>=t?1:0, y=yTrue[i];
      if(y===1){ if(p===1) tp++; else fn++; }
      else     { if(p===1) fp++; else tn++; }
    }
    return {x: fp/(fp+tn)||0, y: tp/(tp+fn)||0};
  }).sort((a,b)=>a.x-b.x);

  let auc=0; for(let i=1;i<pts.length;i++){ const dx=pts[i].x-pts[i-1].x; const my=(pts[i].y+pts[i-1].y)/2; auc+=dx*my; }
  tfvis.render.linechart(document.getElementById('roc-chart'), {values:pts,series:['ROC']}, {xLabel:'FPR',yLabel:'TPR',width:480,height:320});
  const pm=document.getElementById('performance-metrics'); const aucP=pm.querySelector('.auc-line');
  const line=`<p class="auc-line">AUC: ${auc.toFixed(4)}</p>`; if(aucP) aucP.outerHTML=line; else pm.innerHTML+=line;
}

/* ---------------- PREDICT / EXPORT ---------------- */
async function predict(){
  const probs = model.predict(pre.testX).dataSync(); testProbs = probs;
  const out=document.getElementById('prediction-output'); out.innerHTML='<h3>Prediction (first 10)</h3>';
  const t=document.createElement('table'); const h=document.createElement('tr');
  ['PassengerId','Survived','Probability'].forEach(k=>{const th=document.createElement('th'); th.textContent=k; h.appendChild(th);}); t.appendChild(h);
  pre.testIds.slice(0,10).forEach((id,i)=>{ const tr=document.createElement('tr'); tr.innerHTML=`<td>${id}</td><td>${probs[i]>=0.5?1:0}</td><td>${probs[i].toFixed(4)}</td>`; t.appendChild(tr); });
  out.appendChild(t);
  document.getElementById('export-btn').disabled=false;
}

async function exportResults(){
  if(!testProbs){alert('Predict first');return;}
  const sub = ['PassengerId,Survived'].concat(pre.testIds.map((id,i)=>`${id},${testProbs[i]>=0.5?1:0}`)).join('\n');
  const probsCsv = ['PassengerId,Probability'].concat(pre.testIds.map((id,i)=>`${id},${testProbs[i].toFixed(6)}`)).join('\n');
  const dl=(name,text)=>{ const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([text],{type:'text/csv'})); a.download=name; a.click(); };
  dl('submission.csv', sub); dl('probabilities.csv', probsCsv);
  await model.save('downloads://titanic-tfjs-model');
}
