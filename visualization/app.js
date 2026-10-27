(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const canvas = $('scene'), ctx = canvas.getContext('2d');
  const nodeColors = ['#f5f5f5','#facc68','#f194ff','#f16e72','#61bbff'];
  const edgeColors = ['#b3bdcd','#f4b966','#c895f5'];
  let data, frames = [], frameData = null, index = 0;
  let variant = null;
  let selected = -1, projected = [], timer = null, coordinate = 'ee';
  let requestToken = 0, episodeToken = 0, busy = false;
  let yaw = -.68, pitch = .38, scale = 1, panX = 0, panY = 0, focus = [0,0,0];
  const showNode = [true,true,true,true,true], showEdge = [true,true,true];
  const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
  function unpack(frame) {
    const raw=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
    const f32=s=>new Float32Array(raw(s).buffer);
    const i32=s=>new Int32Array(raw(s).buffer);
    const rows=(values,width)=>Array.from({length:values.length/width},(_,i)=>Array.from(values.subarray(i*width,(i+1)*width)));
    const a=i32(frame.edge_index);
    return {t:frame.t,
      node_input:rows(f32(frame.node_input),frame.feature_dimension || 14),
      world_position:rows(f32(frame.world_position),3),
      node_type:Array.from(raw(frame.node_type)),
      edge_index:Array.from({length:frame.e},(_,i)=>[a[i],a[frame.e+i]]),
      edge_attr:rows(f32(frame.edge_attr),4),
      edge_type:Array.from(raw(frame.edge_type)),
      source_index:Array.from(i32(frame.source_index)), node_ids:frame.node_ids, source_label:frame.source_label || 'source index',
      feature_fields:frame.feature_fields || [], phase:frame.phase, timestamp:frame.timestamp};
  }
  function current() {
    return frameData;
  }
  function positions(frame) {
    return coordinate==='world' ? frame.world_position : frame.node_input.map(n=>n.slice(0,3));
  }
  function cameraFit() {
    if (!current()) return;
    const p = positions(current());
    for (let j=0;j<3;j++) focus[j] = (Math.min(...p.map(x=>x[j]))+Math.max(...p.map(x=>x[j])))/2;
    let radius = 0;
    for (const v of p) radius = Math.max(radius,Math.hypot(v[0]-focus[0],v[1]-focus[1],v[2]-focus[2]));
    scale = .42 * Math.min(canvas.clientWidth,canvas.clientHeight) / Math.max(radius,.04);
    panX = panY = 0; draw();
  }
  function project(p) {
    const x=p[0]-focus[0], y=p[1]-focus[1], z=p[2]-focus[2];
    const h=x*Math.cos(yaw)-y*Math.sin(yaw), d=x*Math.sin(yaw)+y*Math.cos(yaw);
    const up=z*Math.cos(pitch)-d*Math.sin(pitch), depth=z*Math.sin(pitch)+d*Math.cos(pitch);
    const perspective=1/(1+clamp(depth*scale/Math.max(canvas.clientWidth,canvas.clientHeight),-.7,2)*.38);
    return {x:canvas.clientWidth/2+panX+h*scale*perspective,
            y:canvas.clientHeight/2+panY-up*scale*perspective, depth, perspective};
  }
  function resize() {
    const dpr=window.devicePixelRatio||1, w=canvas.clientWidth, h=canvas.clientHeight;
    canvas.width=Math.round(w*dpr); canvas.height=Math.round(h*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0); draw();
  }
  function axis(origin, vector, color, label) {
    const a=project(origin), b=project(vector);
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
    ctx.fillStyle=color; ctx.font='bold 12px system-ui'; ctx.fillText(label,b.x+4,b.y-4);
  }
  function draw() {
    const w=canvas.clientWidth,h=canvas.clientHeight;
    ctx.clearRect(0,0,w,h);
    if (!current()) return;
    const frame=current(), nodes=frame.node_input, types=frame.node_type;
    projected=positions(frame).map(p=>project(p));
    const span=Math.min(w,h)/scale*.13;
    axis(focus,[focus[0]+span,focus[1],focus[2]],'#ff7777','X');
    axis(focus,[focus[0],focus[1]+span,focus[2]],'#7bdb99','Y');
    axis(focus,[focus[0],focus[1],focus[2]+span],'#7dabff','Z');
    const origin=project([0,0,0]);
    if (origin.x>=0 && origin.x<=w && origin.y>=0 && origin.y<=h) {
      ctx.strokeStyle='#e9eef8'; ctx.globalAlpha=.65; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(origin.x-5,origin.y); ctx.lineTo(origin.x+5,origin.y);
      ctx.moveTo(origin.x,origin.y-5); ctx.lineTo(origin.x,origin.y+5); ctx.stroke();
      ctx.font='11px system-ui'; ctx.fillStyle='#e9eef8'; ctx.fillText('0,0,0',origin.x+7,origin.y-7);
      ctx.globalAlpha=1;
    }
    const seen=new Set();
    for (let e=0;e<frame.edge_index.length;e++) {
      const type=frame.edge_type[e], pair=frame.edge_index[e], a=pair[0], b=pair[1];
      if (!showEdge[type] || !showNode[types[a]] || !showNode[types[b]]) continue;
      const key=type+':'+Math.min(a,b)+':'+Math.max(a,b);
      if (seen.has(key)) continue;
      seen.add(key);
      ctx.strokeStyle=edgeColors[type]; ctx.globalAlpha=type===0?.16:.62;
      ctx.lineWidth=type===0?1:1.5;
      ctx.beginPath(); ctx.moveTo(projected[a].x,projected[a].y);
      ctx.lineTo(projected[b].x,projected[b].y); ctx.stroke();
    }
    ctx.globalAlpha=1;
    const order=nodes.map((_,i)=>i).sort((a,b)=>projected[b].depth-projected[a].depth);
    for (const i of order) {
      if (!showNode[types[i]]) continue;
      const p=projected[i], robot=types[i]<3, radius=(robot?6:2.8)*clamp(p.perspective,.6,1.7);
      ctx.fillStyle=nodeColors[types[i]];
      if (!robot && $('color-mode').value==='status' && nodes[i].length>14) {
        ctx.fillStyle=nodes[i][14]>.5?'#55d9e6':nodes[i][15]>.5?'#f2a55a':'#d191ff';
      }
      ctx.beginPath(); ctx.arc(p.x,p.y,radius,0,2*Math.PI); ctx.fill();
      if (i===selected) { ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(p.x,p.y,radius+5,0,2*Math.PI); ctx.stroke(); }
      if (robot) { ctx.fillStyle='#f3f6fb'; ctx.font='12px system-ui'; ctx.fillText(data.node_types[types[i]],p.x+9,p.y-7); }
    }
  }
  function updateInspector() {
    if (!current()) { $('inspector').textContent='프레임을 불러오는 중입니다.'; return; }
    const frame=current();
    if (selected<0 || selected>=frame.node_input.length) { $('inspector').innerHTML='점 하나를 클릭하세요.'; return; }
    const n=frame.node_input[selected], t=frame.node_type[selected];
    let incoming=0,outgoing=0, links=[0,0,0], neighbors=[];
    for (let i=0;i<frame.edge_index.length;i++) {
      const [a,b]=frame.edge_index[i];
      if (a===selected) { outgoing++; links[frame.edge_type[i]]++;
        neighbors.push({target:b,type:frame.edge_type[i],attr:frame.edge_attr[i]}); }
      if (b===selected) incoming++;
    }
    const source=selected>=3?(frame.node_ids ? frame.node_ids[selected-3] : frame.source_index[selected-3]):'—';
    const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const vec=a=>a.map(x=>Number(x).toPrecision(6)).join(', ');
    $('inspector').innerHTML=`<strong>#${selected} · ${data.node_types[t]}</strong><br><span class="pill">${frame.node_ids?'node id':escape(frame.source_label)} ${escape(source)}</span>`+
      `<div class="row"><span>in / out</span><strong>${incoming} / ${outgoing}</strong></div>`+
      `<div class="muted">out: ${links.map((v,i)=>data.edge_types[i]+' '+v).join(' · ')}</div>`+
      `<h3>노드 위치 [m]</h3><pre>EE (모델 입력) ${vec(n.slice(0,3))}\n월드 절대     ${vec(frame.world_position[selected])}</pre>`+
      `<h3>나머지 node_input</h3><pre>F5           ${vec(n.slice(3,8))}\navailable    ${vec(n.slice(8,11))}\nvalid        ${vec(n.slice(11,14))}</pre>`+
      (n.length>14?`<h3>관측 상태 · 저장된 입력</h3><pre>${frame.feature_fields.map((name,i)=>escape(name)+'  '+Number(n[14+i]).toPrecision(6)).join('\n')}</pre>`:'')+
      `<h3>Outgoing edge_attr [distance m, direction xyz]</h3><pre>`+
      neighbors.slice(0,12).map(x=>`→ #${x.target} ${data.edge_types[x.type]}\n  ${vec(x.attr)}`).join('\n')+
      (neighbors.length>12?`\n… ${neighbors.length-12} more`:'')+'</pre>';
  }
  async function setFrame(i, fit=false) {
    if (!data || !frames.length) return;
    index=clamp(i,0,frames.length-1); selected=-1;
    const token=++requestToken, episodeId=data.episode.id, timestep=frames[index];
    $('timeline').value=index; $('frame-label').textContent=`t = ${timestep} · 로딩 중`;
    $('frame-count').textContent=`${index+1} / ${frames.length}`;
    try {
      const packed=await getJSON(`/api/frame?id=${encodeURIComponent(episodeId)}&t=${timestep}`+(variant?`&variant=${encodeURIComponent(variant)}`:''));
      if (token!==requestToken) return;
      frameData=unpack(packed);
    } catch(err) {
      if (token===requestToken) $('summary').textContent='프레임 로드 실패: '+err.message;
      return;
    }
    $('frame-label').textContent=`t = ${timestep}`+(current().phase?` · ${current().phase}`:'');
    const f=current(), nc=data.node_types.map((_,t)=>f.node_type.filter(x=>x===t).length),
          ec=data.edge_types.map((_,t)=>f.edge_type.filter(x=>x===t).length);
    $('summary').innerHTML=`<div class="row"><span>노드</span><strong>${f.node_input.length}</strong></div>`+
      nc.map((v,t)=>`<div class="row"><span>${data.node_types[t]}</span><span>${v}</span></div>`).join('')+
      `<div class="row"><span>방향 엣지</span><strong>${f.edge_index.length}</strong></div>`+
      ec.map((v,t)=>`<div class="row"><span>${data.edge_types[t]}</span><span>${v}</span></div>`).join('');
    updateInspector(); if (fit) cameraFit(); else draw();
  }
  function filters(container,names,colors,state) {
    names.forEach((name,i)=>{
      const label=document.createElement('label'); label.className='check';
      const input=document.createElement('input'); input.type='checkbox'; input.checked=true;
      input.addEventListener('change',()=>{state[i]=input.checked;draw();});
      const swatch=document.createElement('span'); swatch.className=state===showNode?'swatch':'line'; swatch.style.background=colors[i];
      label.append(input,swatch,document.createTextNode(name)); container.append(label);
    });
  }
  function stop() { if (timer) clearInterval(timer); timer=null; $('play').textContent='재생'; }
  $('play').onclick=()=>{if(timer){stop();return;} $('play').textContent='일시정지'; timer=setInterval(async()=>{
    if(busy)return; if(index>=frames.length-1){stop();return;}
    busy=true; try {await setFrame(index+1);} finally {busy=false;}
  },200);};
  $('prev').onclick=()=>{stop();setFrame(index-1);}; $('next').onclick=()=>{stop();setFrame(index+1);};
  $('timeline').oninput=e=>{stop();setFrame(Number(e.target.value));}; $('reset').onclick=cameraFit;
  $('graph-variant').onchange=()=>{
    stop(); variant=$('graph-variant').value;
    const url=new URL(location.href);url.searchParams.set('variant',variant);history.replaceState(null,'',url);
    $('variant-note').textContent=variant.startsWith('O')?'Oracle · GT를 사용하는 평가 조건입니다.':'저장된 센서 기반 그래프 · 물체당 최대 128개 노드';
    setFrame(index,true);
  };
  $('color-mode').onchange=draw;
  function setCoordinate(value) {
    coordinate=value;
    $('frame-ee').classList.toggle('primary',value==='ee');
    $('frame-world').classList.toggle('primary',value==='world');
    $('frame-ee').setAttribute('aria-pressed',String(value==='ee'));
    $('frame-world').setAttribute('aria-pressed',String(value==='world'));
    $('coordinate-hint').textContent=(value==='ee'?'EE 상대 좌표':'월드 절대 좌표')+' · 단위 m · X 빨강 / Y 초록 / Z 파랑';
    cameraFit(); updateInspector();
  }
  $('frame-ee').onclick=()=>setCoordinate('ee');
  $('frame-world').onclick=()=>setCoordinate('world');
  let drag=null;
  canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY, moved:false, pan:e.button===2||e.shiftKey};});
  canvas.addEventListener('pointermove',e=>{
    if(!drag)return;
    const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
    if(Math.abs(dx)+Math.abs(dy)>1)drag.moved=true;
    if(drag.pan){panX+=dx;panY+=dy;}else{yaw+=dx*.008;pitch=clamp(pitch+dy*.008,-1.5,1.5);}
    drag.x=e.clientX;drag.y=e.clientY;draw();
  });
  canvas.addEventListener('pointerup',e=>{
    if(!drag)return;
    if(!drag.moved){const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
      let best=-1,dist=100;
      for(let i=0;current() && i<projected.length;i++)if(showNode[current().node_type[i]]){
        const d=(projected[i].x-x)**2+(projected[i].y-y)**2;
        if(d<dist){dist=d;best=i;}
      }
      selected=best;updateInspector();draw();
    }
    drag=null;
  });
  canvas.addEventListener('contextmenu',e=>e.preventDefault());
  canvas.addEventListener('wheel',e=>{e.preventDefault();scale=clamp(scale*Math.exp(-e.deltaY*.001),20,20000);draw();},{passive:false});
  window.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){stop();setFrame(index-1);}if(e.key==='ArrowRight'){stop();setFrame(index+1);}});
  async function getJSON(url) {
    const response=await fetch(url,{cache:'no-store'});
    const value=await response.json();
    if(!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
    return value;
  }
  const labels={success:'성공',failure:'실패',unknown:'미확인'};
  function showOutcome(episode) {
    const status=$('episode-status'); status.className='status '+episode.result;
    status.textContent=labels[episode.result];
    $('episode-meta').textContent=`${episode.frames}프레임`+
      (episode.first_success_step!=null?` · 첫 성공 ${episode.first_success_step} step`:'')+
      (episode.status?` · ${episode.status}`:'');
  }
  async function openEpisode(id) {
    stop(); requestToken++; const token=++episodeToken; frameData=null; draw();
    $('summary').textContent='Episode를 불러오는 중입니다.';
    try {
      const detail=await getJSON(`/api/episode?id=${encodeURIComponent(id)}`);
      if(token!==episodeToken) return;
      data=detail;
      variant=data.variants?(new URLSearchParams(location.search).get('variant') || data.default_variant):null;
      if(data.variants && !data.variants.some(v=>v.id===variant))variant=data.default_variant;
      $('variant-controls').hidden=!data.variants; $('status-controls').hidden=!data.variants;
      if(data.variants){
        $('graph-variant').replaceChildren(...data.variants.map(v=>{const o=document.createElement('option');o.value=v.id;o.textContent=v.label;return o;}));
        $('graph-variant').value=variant;
        $('variant-note').textContent=variant.startsWith('O')?'Oracle · GT를 사용하는 평가 조건입니다.':'저장된 센서 기반 그래프 · 물체당 최대 128개 노드';
        $('capture-link').href='/captures?episode='+encodeURIComponent(data.episode.capture_id);
      }
      frames=data.timesteps; index=0; selected=-1;
      $('episode').textContent=data.episode.id;
      showOutcome(data.episode);
      if(data.episode.kind==='persistent_geometry')$('episode-meta').textContent+=` · ${data.episode.split} · ${data.episode.revision}`;
      $('node-filters').replaceChildren(); $('edge-filters').replaceChildren();
      showNode.fill(true); showEdge.fill(true);
      filters($('node-filters'),data.node_types,nodeColors,showNode);
      filters($('edge-filters'),data.edge_types,edgeColors,showEdge);
      $('timeline').max=frames.length-1;
      await setFrame(0,true);
    } catch(err) { if(token===episodeToken) $('summary').textContent='Episode 로드 실패: '+err.message; }
  }
  function start() {
    resize(); new ResizeObserver(resize).observe(canvas);
    const id=new URLSearchParams(location.search).get('episode');
    if (id) openEpisode(id);
    else location.replace('/');
  }
  start();
})();
