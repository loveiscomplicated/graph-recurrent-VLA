'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {URL, Request} = globalThis;
let observer, fetched;
const location = new URL('https://private.static.hf.space/index.html?__sign=test-signature&episode=123');
const document = {baseURI: location.href};
const context = {URL, Request, location, document,
  window: {fetch: (input) => { fetched = input; }},
  MutationObserver: class { constructor(cb) { observer=cb; } observe() {} }};
vm.runInNewContext(fs.readFileSync('visualization/static_auth.js', 'utf8'), context);
function element(attrs) {
  return {nodeType: 1, getAttribute: name => attrs[name] || null,
    setAttribute: (name,value) => { attrs[name]=value; }, attrs};
}
const local=element({href:'./results.html'}), remote=element({href:'https://example.org/report'}), img=element({src:'./plot.png'});
observer([{type:'childList', addedNodes:[local, remote, img]}]);
assert.equal(new URL(local.attrs.href).searchParams.get('__sign'), 'test-signature');
assert.equal(new URL(local.attrs.href).searchParams.has('episode'), false);
assert.equal(remote.attrs.href, 'https://example.org/report');
assert.equal(new URL(img.attrs.src).searchParams.get('__sign'), 'test-signature');
context.window.fetch('./data/results.json');
assert.equal(new URL(fetched).searchParams.get('__sign'), 'test-signature');
context.window.fetch('https://example.org/data');
assert.equal(fetched, 'https://example.org/data');
console.log('Private static Space navigation and resource signing passed');
