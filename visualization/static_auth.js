// Private Hugging Face Static Spaces authenticate each URL independently.
// Keep their short-lived signature on same-origin resources and navigation only.
(() => {
  'use strict';
  const signature = new URL(location.href).searchParams.get('__sign');
  if (!signature) return;
  function signed(value) {
    if (!value) return value;
    try {
      const url = new URL(value, document.baseURI);
      if (url.origin !== location.origin || !/^https?:$/.test(url.protocol)) return value;
      if (url.searchParams.get('__sign') === signature) return value;
      url.searchParams.set('__sign', signature);
      return url.href;
    } catch (_) { return value; }
  }
  function update(element) {
    if (element.nodeType !== 1) return;
    for (const name of ['src', 'href', 'poster']) {
      const value = element.getAttribute(name);
      if (value) {
        const replacement = signed(value);
        if (replacement !== value) element.setAttribute(name, replacement);
      }
    }
  }
  const observer = new MutationObserver(records => {
    for (const record of records) {
      if (record.type === 'attributes') update(record.target);
      for (const node of record.addedNodes) {
        update(node);
        if (node.querySelectorAll) node.querySelectorAll('[src],[href],[poster]').forEach(update);
      }
    }
  });
  observer.observe(document, {subtree: true, childList: true, attributes: true,
    attributeFilter: ['src', 'href', 'poster']});
  const originalFetch = window.fetch;
  if (originalFetch) window.fetch = function(input, init) {
    if (typeof input === 'string' || input instanceof URL) input = signed(String(input));
    else if (input instanceof Request) input = new Request(signed(input.url), input);
    return originalFetch.call(this, input, init);
  };
})();
