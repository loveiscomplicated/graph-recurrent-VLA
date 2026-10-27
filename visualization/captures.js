(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const phases = {reach:'접근', descend:'하강', close:'잡기', lift:'들기', align:'정렬', insert:'삽입'};
  let episodes = [], episode = null, frame = null, index = 0, generation = 0, playing = false, timer = null;
  const views = new Map(), cache = new Map();
  let mediaRuns = [], mediaGeneration = 0, videoFps = 20;
  const raw = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));

  async function json(url) {
    const dataRoot = document.documentElement.dataset.captureData;
    if (dataRoot) {
      const request = new URL(url, location.href), id = request.searchParams.get('id');
      const episodePath = id?.split('/').map(encodeURIComponent).join('/');
      if (request.pathname === '/api/captures') url = `${dataRoot}/captures.json`;
      else if (request.pathname === '/api/capture-media') url = `${dataRoot}/${episodePath}/media.json`;
      else if (request.pathname === '/api/capture-frame') url = `${dataRoot}/${episodePath}/frames/${request.searchParams.get('t')}.json`;
      else throw new Error('지원하지 않는 정적 데이터 요청');
    }
    const response = await fetch(url, {cache:'no-store'}), value = await response.json();
    if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
    return value;
  }
  function stop() { playing = false; clearTimeout(timer); $('play').textContent = '재생'; }
  function fail(error) { stop(); $('capture-error').textContent = '로드 실패: '+error.message; }

  function image(canvas, camera, kind) {
    const n = camera.width * camera.height, pixels = new Uint8ClampedArray(n * 4);
    const limit = Number($('depth-limit').value);
    for (let i = 0; i < n; i++) {
      let color = Array.from(camera.rgb.subarray(i*3, i*3+3));
      if (kind === 'depth') {
        const value = camera.depth[i], f = Math.min(1, Math.max(0, value / limit));
        color = value > 0 ? [255-205*f, 220-40*f, 80+175*f] : [0,0,0];
      } else if (kind !== 'rgb') {
        const mask = camera[kind], a = mask[i], b = mask[n+i];
        const tint = a && b ? [237,149,255] : a ? [255,112,121] : b ? [101,186,255] : null;
        if (tint) color = color.map((x,j) => Math.round(x*.38+tint[j]*.62));
      }
      pixels.set([...color,255], i*4);
    }
    canvas.getContext('2d').putImageData(new ImageData(pixels, camera.width, camera.height), 0, 0);
    canvas.title = kind === 'depth' ? `깊이: 0–${limit} mm; ≤0 무효` : `${camera.name} · ${kind}`;
  }

  function draw(value) {
    frame = value;
    const rows = [];
    for (const camera of value.cameras) {
      if (!views.has(camera.name)) {
        const section = document.createElement('section'); section.className = 'capture-camera';
        const title = document.createElement('h2'); title.textContent = `${camera.role === 'main' ? 'Main' : 'Wrist'} · ${camera.name}`;
        const images = document.createElement('div'); images.className = 'capture-images';
        const canvases = {};
        for (const [kind, label] of [['rgb','RGB'],['depth','깊이 · mm'],['predicted','예측 마스크'],['gt','GT 마스크 · 평가용']]) {
          const figure = document.createElement('figure'), caption = document.createElement('figcaption'), canvas = document.createElement('canvas');
          caption.textContent = label; canvas.width = camera.width; canvas.height = camera.height;
          canvas.setAttribute('aria-label', `${camera.name} ${label}`);
          figure.append(caption, canvas); images.append(figure); canvases[kind] = canvas;
        }
        section.append(title, images); $('camera-views').append(section); views.set(camera.name, canvases);
      }
      for (const [kind, canvas] of Object.entries(views.get(camera.name))) image(canvas, camera, kind);
      for (const object of camera.objects) {
        const tr = document.createElement('tr');
        const range = camera.depth_min_mm == null ? '—' : `${camera.depth_min_mm}–${camera.depth_max_mm} mm`;
        for (const text of [camera.name, `${(camera.valid_depth_fraction*100).toFixed(1)}% / ${range}`, object.name, object.predicted_pixels, object.gt_pixels, object.iou == null ? '—' : object.iou.toFixed(3)]) {
          const td = document.createElement('td'); td.textContent = text; tr.append(td);
        }
        rows.push(tr);
      }
    }
    $('capture-stats').replaceChildren(...rows);
    $('frame-status').textContent = `${index+1} / ${episode.frames}프레임 · t=${value.t} · ${value.timestamp.toFixed(2)} s · ${phases[value.phase] || value.phase} (${value.phase}) · 원본 3개 파일 해시 확인됨`;
    $('timeline').value = index;
    $('prev').disabled = index === 0; $('next').disabled = index === episode.frames-1;
    $('phase-jumps').querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.phase === value.phase));
  }

  async function loadFrame(nextIndex) {
    if (!episode) return;
    const token = ++generation;
    index = Math.max(0, Math.min(episode.frames-1, nextIndex));
    const key = `${episode.id}:${episode.timesteps[index]}`;
    $('capture-error').textContent = '';
    $('frame-status').textContent = `프레임 ${index+1} / ${episode.frames} 불러오는 중…`;
    try {
      let value = cache.get(key);
      if (!value) {
        value = await json(`/api/capture-frame?id=${encodeURIComponent(episode.id)}&t=${episode.timesteps[index]}`);
        value.cameras.forEach(c => {
          c.rgb = raw(c.rgb); c.predicted = raw(c.predicted); c.gt = raw(c.gt);
          const bytes = raw(c.depth_mm), data = new DataView(bytes.buffer);
          c.depth = Int16Array.from({length:c.width*c.height}, (_,i) => data.getInt16(i*2, true));
        });
        cache.set(key, value);
        if (cache.size > 24) cache.delete(cache.keys().next().value);
      }
      if (token !== generation) return;
      draw(value);
      if (playing) {
        if (index === episode.frames-1) stop();
        else timer = setTimeout(() => loadFrame(index+1), 1000/Number($('speed').value));
      }
    } catch (error) { if (token === generation) fail(error); }
  }

  function select() {
    stop(); ++generation; cache.clear(); views.clear(); frame = null;
    $('camera-views').replaceChildren(); $('capture-stats').replaceChildren();
    episode = episodes.find(e => e.id === $('capture-select').value);
    if (!episode) return;
    loadMedia(episode.id);
    const url = new URL(location.href); url.searchParams.set('episode', episode.id); history.replaceState(null, '', url);
    $('capture-meta').textContent = `${episode.run_id} · seed ${episode.seed} · ${episode.split} · ${episode.frames}프레임 · ${episode.complete ? '수집 종료' : '수집 미완료'} · ${episode.source_kind}`;
    $('timeline').max = episode.frames-1;
    for (const id of ['timeline','play','prev','next']) $(id).disabled = false;
    const buttons = [];
    episode.phases.forEach((phase, i) => {
      if (i > 0 && episode.phases[i-1] === phase) return;
      const b = document.createElement('button'); b.textContent = `${phases[phase] || phase} · t${episode.timesteps[i]}`; b.dataset.phase = phase;
      b.addEventListener('click', () => { stop(); loadFrame(i); }); buttons.push(b);
    });
    $('phase-jumps').replaceChildren(...buttons);
    loadFrame(0);
  }
  function showVideoVariant() {
    const run=mediaRuns.find(r=>r.run_id===$('video-run').value);
    const variant=run?.variants.find(v=>v.id===$('video-variant').value);
    if(!variant)return;
    for(const [kind,id] of [['graph_rgb','video-graph-rgb'],['graph_only','video-graph-only']]){
      const video=$(id);video.pause();video.src=variant.videos[kind].url;video.load();
    }
    videoFps=variant.videos.graph_only.fps;
    $('video-meta').textContent=`${run.revision} · ${videoFps.toFixed(0)} fps · 전체 ${variant.videos.graph_only.frames}프레임 · ${variant.label}`;
    const isPointcloud=variant.kind==='pointcloud';
    $('video-description').textContent=variant.description || '';
    $('video-description').hidden=!variant.description;
    $('video-caption-rgb').textContent=isPointcloud?'a. Point cloud + RGB · Main / Wrist':'a. Graph + RGB · Main / Wrist';
    $('video-caption-only').textContent=isPointcloud?'b. Only point cloud':'b. Only graph';
    $('video-graph-rgb').setAttribute('aria-label',isPointcloud?'포인트 클라우드와 RGB 영상':'그래프와 RGB 영상');
    $('video-graph-only').setAttribute('aria-label',isPointcloud?'포인트 클라우드만 영상':'그래프만 영상');
    if ($('interactive-graph')) {
      $('interactive-graph').hidden=isPointcloud;
      if(!isPointcloud) $('interactive-graph').href=`${document.documentElement.dataset.captureData?'./viewer.html':'/viewer'}?episode=${encodeURIComponent(run.geometry_id)}&variant=${encodeURIComponent(variant.id)}`;
    }
  }
  function showVideoRun(){
    const run=mediaRuns.find(r=>r.run_id===$('video-run').value);
    if(!run)return;
    $('video-variant').replaceChildren(...run.variants.map(v=>{const o=document.createElement('option');o.value=v.id;o.textContent=v.label;return o;}));
    if(run.variants.some(v=>v.id==='G4'))$('video-variant').value='G4';
    showVideoVariant();
  }
  async function loadMedia(id){
    const token=++mediaGeneration;
    $('capture-videos').hidden=true;
    for(const name of ['video-graph-rgb','video-graph-only']){const v=$(name);v.pause();v.removeAttribute('src');v.load();}
    $('video-message').textContent='그래프 영상 확인 중…';
    try{
      const value=await json('/api/capture-media?id='+encodeURIComponent(id));
      if(token!==mediaGeneration)return;
      mediaRuns=value.runs;
      if(!mediaRuns.length){$('video-message').textContent='이 에피소드에 연결된 그래프 영상이 없습니다.';return;}
      $('video-run').replaceChildren(...mediaRuns.map(r=>{const o=document.createElement('option');o.value=r.run_id;o.textContent=`${r.revision} · ${r.run_id.slice(0,8)}`;return o;}));
      $('capture-videos').hidden=false;$('video-message').textContent='';showVideoRun();
    }catch(error){if(token===mediaGeneration)$('video-message').textContent='그래프 영상 로드 실패: '+error.message;}
  }
  $('video-run').addEventListener('change',showVideoRun);
  $('video-variant').addEventListener('change',showVideoVariant);
  $('video-seek').addEventListener('click',()=>{
    for(const name of ['video-graph-rgb','video-graph-only']){const v=$(name);v.pause();if(v.readyState>0)v.currentTime=index/videoFps;}
  });
  $('capture-select').addEventListener('change', select);
  $('timeline').addEventListener('input', () => { stop(); loadFrame(Number($('timeline').value)); });
  $('prev').addEventListener('click', () => { stop(); loadFrame(index-1); });
  $('next').addEventListener('click', () => { stop(); loadFrame(index+1); });
  $('play').addEventListener('click', () => {
    if (playing) { stop(); return; }
    playing = true; $('play').textContent = '일시정지'; loadFrame(index === episode.frames-1 ? 0 : index);
  });
  $('depth-limit').addEventListener('change', () => { if (frame) draw(frame); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  json('/api/captures').then(value => {
    episodes = value.episodes;
    if (!episodes.length) { $('capture-select').replaceChildren(); $('frame-status').textContent = '수집 데이터가 없습니다. README의 로컬 복사 경로를 확인하세요.'; return; }
    $('capture-select').replaceChildren(...episodes.map(e => {
      const option = document.createElement('option'); option.value = e.id;
      option.textContent = `seed ${e.seed} · ${e.split} · ${e.frames}프레임 · ${e.run_id}`; return option;
    }));
    const requested = new URLSearchParams(location.search).get('episode');
    if (episodes.some(e => e.id === requested)) $('capture-select').value = requested;
    $('capture-select').disabled = false; select();
  }).catch(fail);
})();
