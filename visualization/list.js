(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const labels = {success:'성공', failure:'실패', unknown:'미확인'};
  let episodes = [], result = 'all';

  async function load() {
    $('list-summary').textContent = '목록을 불러오는 중입니다.';
    try {
      const response = await fetch('/api/episodes', {cache:'no-store'});
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
      episodes = value.episodes;
      for (const kind of ['all', ...Object.keys(labels)]) {
        $('count-'+kind).textContent = kind === 'all' ? episodes.length : episodes.filter(e=>e.result===kind).length;
      }
      render();
    } catch (error) {
      $('list-summary').textContent = '목록 로드 실패: '+error.message;
      $('episode-cards').replaceChildren();
    }
  }

  function render() {
    const needle = $('episode-search').value.trim().toLowerCase();
    const shown = episodes.filter(e => (result === 'all' || e.result === result) &&
      (!needle || e.id.toLowerCase().includes(needle)));
    $('list-summary').textContent = `표시 ${shown.length}개 · 전체 ${episodes.length}개`;
    const cards = shown.map(e => {
      const card = document.createElement('a');
      card.className = 'episode-card';
      card.href = '/viewer?episode='+encodeURIComponent(e.id);
      const top = document.createElement('div'); top.className = 'card-top';
      const name = document.createElement('strong'); name.className = 'card-name'; name.textContent = e.name;
      const badge = document.createElement('span'); badge.className = 'status '+e.result;
      badge.textContent = labels[e.result] || labels.unknown;
      top.append(name, badge);
      const path = document.createElement('div'); path.className = 'card-path'; path.textContent = e.id;
      const meta = document.createElement('div'); meta.className = 'card-meta';
      const parts = [`${e.frames}프레임`, `t ${e.first_timestep}–${e.last_timestep}`];
      if (e.first_success_step != null) parts.push(`첫 성공 ${e.first_success_step} step`);
      if (e.status) parts.push(e.status);
      meta.textContent = parts.join(' · ');
      const action = document.createElement('span'); action.className = 'card-action';
      action.textContent = '3D 그래프 열기 →';
      card.append(top, path, meta, action);
      return card;
    });
    $('episode-cards').replaceChildren(...cards);
    if (!shown.length) {
      const empty = document.createElement('p'); empty.className = 'empty-list';
      empty.textContent = episodes.length ? '조건에 맞는 rollout이 없습니다.' : '볼 수 있는 rollout graph가 없습니다.';
      $('episode-cards').append(empty);
    }
  }

  $('episode-search').addEventListener('input', render);
  $('refresh').addEventListener('click', load);
  document.querySelectorAll('[data-result]').forEach(button => button.addEventListener('click', () => {
    result = button.dataset.result;
    document.querySelectorAll('[data-result]').forEach(item => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    render();
  }));
  const legacy = new URLSearchParams(location.search).get('episode');
  if (legacy) location.replace('/viewer?episode='+encodeURIComponent(legacy));
  else load();
})();
