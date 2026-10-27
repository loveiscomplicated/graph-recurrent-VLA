// Functional UI checks without a browser, network or extra dependencies.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

class Element {
  constructor(tag='div') { this.tagName=tag.toUpperCase(); this.children=[]; this.listeners={}; this.dataset={}; this.value=''; this.textContent=''; this.hidden=false; this.attributes={}; this.classList={toggle(){}}; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children=children; }
  setAttribute(key,value) { this.attributes[key]=value; }
  addEventListener(key,fn) { this.listeners[key]=fn; }
  querySelectorAll() { return this.children.filter(c=>c.tagName==='BUTTON'); }
  focus() {}
}
const data = JSON.parse(fs.readFileSync(process.argv[2]||'outputs/results-main/data/results.json','utf8'));
async function setup(search='') {
  const elements=new Map();
  const get=id=>{if(!elements.has(id)) elements.set(id,new Element()); return elements.get(id);};
  get('result-sort').value='task';
  const all=new Element('button');all.dataset.category='all';get('category-filters').append(all);
  const location={href:'https://example.test/site/index.html'+search,search,replace(value){this.redirect=value;}};
  const context={URL,URLSearchParams,Intl,document:{getElementById:get,createElement:tag=>new Element(tag),documentElement:{dataset:{resultsData:'./data/results.json'}},activeElement:new Element('body'),addEventListener(){}},
    location,history:{replaceState(_a,_b,url){location.href=String(url);}},fetch:async()=>({ok:true,json:async()=>data})};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'results.js'),'utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  return {get,location};
}
(async()=>{
  const {get}=await setup();
  assert.equal(get('result-list').children.length,data.groups.length);
  get('result-search').value='Raw EDE';get('result-search').listeners.input();
  assert.equal(get('result-list').children.length,1);
  assert.match(get('result-summary').textContent,/1개 주제/);
  get('result-search').value='';get('result-search').listeners.input();
  const memory=get('category-filters').children.find(b=>b.dataset.category==='memory');
  get('category-filters').listeners.click({target:{closest:()=>memory}});
  assert.equal(get('result-list').children.length,3);
  get('result-search').value='검색 결과가 없는 문장';get('result-search').listeners.input();
  assert.match(get('result-list').children[0].textContent,/조건에 맞는 결과가 없습니다/);
  const restored=await setup('?q=Raw+EDE');
  assert.equal(restored.get('result-list').children.length,1);
  const legacy=await setup('?episode=pilot%2Fcollection%2Fseed30002');
  assert.match(legacy.location.redirect,/captures.html\?episode=/);
  console.log('PASS: list, keyword search, category filter, empty state, query restore, old capture bookmark');
})().catch(error=>{console.error(error);process.exitCode=1;});
