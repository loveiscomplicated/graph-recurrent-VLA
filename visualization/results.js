(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  let catalog=null, category='all';
  const params=new URLSearchParams(location.search);
  // Preserve old capture bookmarks after the deployed root becomes this page.
  if(document.documentElement.dataset.resultsData && params.has('episode')){
    location.replace('./captures.html'+location.search);return;
  }
  const state=e=>e.status==='running'?'진행 중':e.status==='failed'?'실행 실패 기록':e.status==='finished'?'실행 완료':'실행 상태 미기록';
  function date(value){
    if(!value)return '';
    const time=new Date(value);
    return Number.isNaN(time.getTime())?'':new Intl.DateTimeFormat('ko-KR',{timeZone:'Asia/Seoul',year:'numeric',month:'2-digit',day:'2-digit'}).format(time);
  }
  function element(tag,className,text){const e=document.createElement(tag);if(className)e.className=className;if(text!=null)e.textContent=text;return e;}
  function row(group){
    const entry=group.primary,item=element('article','result-item');
    const marker=element('div','result-marker');marker.append(element('span','task-code',group.task),element('span','category-label',catalog.categories[group.category]));
    const copy=element('div','result-copy'),heading=element('h3'),title=entry.url?element('a',null,group.title):element('span',null,group.title);
    if(entry.url)title.href=entry.url;
    heading.append(title);copy.append(heading,element('p','result-subtitle',group.subtitle));
    const meta=element('div','result-meta');meta.append(element('span','state',state(entry)));
    if(date(entry.date))meta.append(element('time',null,date(entry.date)));
    if(entry.notice)meta.append(element('span','notice',entry.notice));
    copy.append(meta);
    if(group.versions.length>1){
      const details=element('details','versions'),summary=element('summary',null,`관련 실행 ${group.versions.length}개`),list=element('ul','version-list');
      for(const version of group.versions){
        const li=element('li'),a=element(version.url?'a':'span',null,`${version.run_id===entry.run_id?'대표 보고서 · ':''}${date(version.date)||'날짜 미기록'} · ${state(version)}`);
        if(version.url)a.href=version.url;
        li.append(a,element('span','run-code',version.run_id.slice(0,8)));
        if(version.notice)li.append(element('span',null,version.notice));list.append(li);
      }
      details.append(summary,list);copy.append(details);
    }
    const open=element(entry.url?'a':'span','result-open',entry.url?'보고서 보기':'기록 진행 중');
    if(entry.url){open.href=entry.url;open.setAttribute('aria-label',`${group.title} 보고서 보기`);open.append(element('span',null,'↗'));}
    item.append(marker,copy,open);return item;
  }
  function render(){
    const query=$('result-search').value.trim().toLocaleLowerCase('ko');
    let groups=catalog.groups.filter(g=>(category==='all'||g.category===category)&&[g.title,g.subtitle,g.task,catalog.categories[g.category],g.id,...g.versions.map(e=>e.run_id)].join(' ').toLocaleLowerCase('ko').includes(query));
    if($('result-sort').value==='date')groups=groups.slice().sort((a,b)=>(b.primary.date||'').localeCompare(a.primary.date||''));
    const rows=groups.map(row);
    if(!rows.length)rows.push(element('div','empty-results','조건에 맞는 결과가 없습니다. 다른 주제나 검색어를 선택해 주세요.'));
    $('result-list').replaceChildren(...rows);
    $('result-summary').textContent=`${groups.length}개 주제 · 전체 ${catalog.groups.length}개`;
    for(const button of $('category-filters').querySelectorAll('button')){const active=button.dataset.category===category;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));}
    const url=new URL(location.href);if(query)url.searchParams.set('q',$('result-search').value.trim());else url.searchParams.delete('q');if(category!=='all')url.searchParams.set('category',category);else url.searchParams.delete('category');history.replaceState(null,'',url);
  }
  fetch(document.documentElement.dataset.resultsData||'/api/results',{cache:'no-store'}).then(async response=>{
    if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json();
  }).then(value=>{
    catalog=value;
    $('topic-total').textContent=value.groups.length;
    $('report-total').textContent=`HTML 보고서 ${value.reports}개 · 웹 열람 ${value.available_reports}개`;
    for(const id of ['capture-nav','capture-shortcut'])$(id).href=value.capture_url;
    for(const [key,label] of Object.entries(value.categories)){
      const count=value.groups.filter(g=>g.category===key).length;if(!count)continue;
      const b=element('button',null,label);b.type='button';b.dataset.category=key;b.setAttribute('aria-pressed','false');b.append(element('span','filter-count',count));$('category-filters').append(b);
    }
    $('result-search').value=params.get('q')||'';
    category=value.categories[params.get('category')]?params.get('category'):'all';
    render();
  }).catch(error=>{$('result-summary').textContent='보고서 목록을 불러오지 못했습니다.';$('result-error').hidden=false;$('result-error').textContent=`페이지를 새로고침해 주세요. (${error.message})`;});
  $('category-filters').addEventListener('click',event=>{const b=event.target.closest('button[data-category]');if(b&&catalog){category=b.dataset.category;render();}});
  $('result-search').addEventListener('input',()=>{if(catalog)render();});$('result-sort').addEventListener('change',()=>{if(catalog)render();});
  document.addEventListener('keydown',event=>{if(event.key==='/'&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){event.preventDefault();$('result-search').focus();}});
})();
