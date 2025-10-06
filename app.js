// === Titanic TensorFlow.js Classifier ===
let trainData = null, testData = null;
let pre = {trainX:null, trainY:null, testX:null, testIds:[]};
let model = null, valX=null, valY=null, valProbs=null, testProbs=null;

const TARGET = 'Survived';
const ID = 'PassengerId';

document.getElementById('load-bundled-btn').onclick = loadBundled;
document.getElementById('load-data-btn').onclick = loadUploaded;
document.getElementById('inspect-btn').onclick = inspectData;
document.getElementById('preprocess-btn').onclick = preprocessData;
document.getElementById('create-model-btn').onclick = createModel;
document.getElementById('train-btn').onclick = trainModel;
document.getElementById('predict-btn').onclick = predict;
document.getElementById('export-btn').onclick = exportResults;

function parseCSV(text){
  text=text.replace(/^\uFEFF/,'').replace(/\r\n/g,'\n').trim();
  const lines=text.split('\n');
  const split=line=>{
    const out=[];let cur='',inQ=false;
    for(let i=0;i<line.length;i++){
      const ch=line[i];
      if(ch==='"'){if(inQ&&line[i+1]==='"'){cur+='"';i++;}else inQ=!inQ;}
      else if(ch===','&&!inQ){out.push(cur);cur='';}
      else cur+=ch;
    }
    out.push(cur);return out;
  };
  const headers=split(lines[0]);
  const rows=[];
  for(let li=1;li<lines.length;li++){
    if(!lines[li])continue;
    const vals=split(lines[li]);
    const obj={};
    headers.forEach((h,i)=>{
      let v=vals[i]??null;
      if(v!==null&&!isNaN(v)&&v.trim()!=='')v=parseFloat(v);
      obj[h]=v;
    });
    rows.push(obj);
  }
  return rows;
}

async function loadBundled(){
  const status=document.getElementById('data-status');
  status.textContent='Loading bundled CSVs...';
  try{
    const t=document.getElementById('bundled-train').textContent;
    const s=document.getElementById('bundled-test').textContent;
    trainData=parseCSV(t); testData=parseCSV(s);
    status.textContent=`Bundled data loaded. Train: ${trainData.length}, Test: ${testData.length}`;
    document.getElementById('inspect-btn').disabled=false;
  }catch(e){status.textContent='Error loading embedded CSV';console.error(e);}
}

function readFile(f){return new Promise((res,rej)=>{const r=new FileReader();r.onload=e=>res(e.target.result);r.onerror=()=>rej();r.readAsText(f);});}
async function loadUploaded(){
  const tr=document.getElementById('train-file').files[0];
  const te=document.getElementById('test-file').files[0];
  if(!tr||!te){alert('Upload both files');return;}
  const status=document.getElementById('data-status');status.textContent='Loading...';
  try{
    const [a,b]=await Promise.all([readFile(tr),readFile(te)]);
    trainData=parseCSV(a); testData=parseCSV(b);
    status.textContent=`Uploaded data loaded. Train: ${trainData.length}, Test: ${testData.length}`;
    document.getElementById('inspect-btn').disabled=false;
  }catch(e){status.textContent='Error';}
}

function tableFromObjects(arr){
  const t=document.createElement('table');const h=document.createElement('tr');
  Object.keys(arr[0]).forEach(k=>{const th=document.createElement('th');th.textContent=k;h.appendChild(th);});
  t.appendChild(h);
  arr.forEach(r=>{const tr=document.createElement('tr');
    Object.keys(arr[0]).forEach(k=>{const td=document.createElement('td');td.textContent=r[k];tr.appendChild(td);});
    t.appendChild(tr);});
  return t;
}

function inspectData(){
  const prev=document.getElementById('data-preview');prev.innerHTML='<h3>Preview</h3>';
  prev.appendChild(tableFromObjects(trainData.slice(0,10)));
  const bySex={},byCls={};
  trainData.forEach(r=>{
    if(r.Sex!=null){bySex[r.Sex]??={s:0,t:0};bySex[r.Sex].t++;if(r.Survived==1)bySex[r.Sex].s++;}
    if(r.Pclass!=null){byCls[r.Pclass]??={s:0,t:0};byCls[r.Pclass].t++;if(r.Survived==1)byCls[r.Pclass].s++;}
  });
  const sex=Object.entries(bySex).map(([k,v])=>({index:k,value:v.s/v.t*100}));
  const cls=Object.entries(byCls).map(([k,v])=>({index:'Class '+k,value:v.s/v.t*100}));
  tfvis.render.barchart(document.getElementById('chart-sex'),sex,{xLabel:'Sex',yLabel:'Survival %'});
  tfvis.render.barchart(document.getElementById('chart-pclass'),cls,{xLabel:'Pclass',yLabel:'Survival %'});
  document.getElementById('preprocess-btn').disabled=false;
}

function median(a){a=a.filter(x=>x!=null).sort((a,b)=>a-b);if(!a.length)return 0;const h=Math.floor(a.length/2);return a.length%2?a[h]:(a[h-1]+a[h])/2;}
function std(a){a=a.filter(x=>x!=null);if(!a.length)return 1;const m=a.reduce((s,x)=>s+x,0)/a.length;return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/a.length);}
function mode(a){const f={};a.filter(x=>x!=null).forEach(x=>f[x]=(f[x]||0)+1);return Object.entries(f).sort((a,b)=>b[1]-a[1])[0][0];}
function oneHot(v,c){const a=new Array(c.length).fill(0);const i=c.indexOf(v);if(i>=0)a[i]=1;return a;}

function preprocessData(){
  const ageM=median(trainData.map(r=>r.Age)),fareM=median(trainData.map(r=>r.Fare));
  const ageS=std(trainData.map(r=>r.Age)),fareS=std(trainData.map(r=>r.Fare));
  const embM=mode(trainData.map(r=>r.Embarked));
  const add=document.getElementById('add-family-features').checked;
  const build=r=>{
    const age=(r.Age??ageM-ageM)/ageS,fare=(r.Fare??fareM-fareM)/fareS;
    let f=[age,fare,r.SibSp??0,r.Parch??0];
    f=f.concat(oneHot(r.Pclass,[1,2,3]),oneHot(r.Sex,['male','female']),oneHot(r.Embarked??embM,['C','Q','S']));
    if(add){const fs=(r.SibSp??0)+(r.Parch??0)+1;f.push(fs,fs==1?1:0);}
    return f;
  };
  const X=trainData.map(build),y=trainData.map(r=>r.Survived);
  pre.trainX=tf.tensor2d(X);pre.trainY=tf.tensor1d(y,'float32');
  pre.testX=tf.tensor2d(testData.map(build));pre.testIds=testData.map(r=>r.PassengerId);
  document.getElementById('create-model-btn').disabled=false;
}

function createModel(){
  model=tf.sequential();
  model.add(tf.layers.dense({units:16,activation:'relu',inputShape:[pre.trainX.shape[1]]}));
  model.add(tf.layers.dense({units:1,activation:'sigmoid'}));
  model.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});
  document.getElementById('train-btn').disabled=false;
}

async function trainModel(){
  const n=pre.trainX.shape[0],split=Math.floor(n*0.8),f=pre.trainX.shape[1];
  const trX=pre.trainX.slice([0,0],[split,f]),trY=pre.trainY.slice([0],[split]);
  valX=pre.trainX.slice([split,0],[n-split,f]);valY=pre.trainY.slice([split],[n-split]);
  await model.fit(trX,trY,{epochs:50,batchSize:32,validationData:[valX,valY],
    callbacks:tfvis.show.fitCallbacks({name:'Train'},['loss','acc','val_loss','val_acc'])});
  valProbs=model.predict(valX).dataSync();
  document.getElementById('threshold-slider').disabled=false;
  updateMetrics();document.getElementById('predict-btn').disabled=false;
}

function updateMetrics(){
  const thr=parseFloat(document.getElementById('threshold-slider').value);
  document.getElementById('threshold-value').textContent=thr.toFixed(2);
  const y=Array.from(valY.dataSync()),p=Array.from(valProbs);
  let tp=0,tn=0,fp=0,fn=0;for(let i=0;i<p.length;i++){const pr=p[i]>=thr?1:0;if(pr&&y[i])tp++;else if(!pr&&!y[i])tn++;else if(pr&&!y[i])fp++;else fn++;}
  const acc=(tp+tn)/(tp+tn+fp+fn),prec=tp/(tp+fp)||0,rec=tp/(tp+fn)||0,f1=2*(prec*rec)/(prec+rec)||0;
  document.getElementById('performance-metrics').innerHTML=`<p>Acc ${(acc*100).toFixed(2)}%</p><p>P ${prec.toFixed(3)}</p><p>R ${rec.toFixed(3)}</p><p>F1 ${f1.toFixed(3)}</p>`;
}

async function predict(){
  const probs=model.predict(pre.testX).dataSync();testProbs=probs;
  const out=document.getElementById('prediction-output');out.innerHTML='<h3>Predictions</h3>';
  const t=document.createElement('table');const h=document.createElement('tr');
  ['PassengerId','Survived','Prob'].forEach(k=>{const th=document.createElement('th');th.textContent=k;h.appendChild(th);});t.appendChild(h);
  pre.testIds.forEach((id,i)=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${id}</td><td>${probs[i]>=0.5?1:0}</td><td>${probs[i].toFixed(3)}</td>`;t.appendChild(tr);});
