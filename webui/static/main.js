const $ = s => document.querySelector(s);
const DEFAULT_MODEL = (window.APP_CONFIG && window.APP_CONFIG.DEFAULT_MODEL) || 'mistral-7b.gguf';
const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.DEFAULT_API_BASE) || ''; // same-origin
const STORAGE_KEY = 'local-llm-chat-history';
let history = [];
let sending = false;

function renderMarkdownToHtml(md){
  // Basic markdown rendering with marked
  const html = marked.parse(md || '');
  const wrapper = document.createElement('div');
  wrapper.innerHTML = html;
  // Syntax highlight all code blocks and add copy buttons
  wrapper.querySelectorAll('pre code').forEach(code => {
    hljs.highlightElement(code);
    const pre = code.closest('pre');
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.addEventListener('click', async () => {
      const raw = code.innerText;
      try {
        await navigator.clipboard.writeText(raw);
        btn.textContent = 'Copied!';
        setTimeout(()=>btn.textContent='Copy', 1200);
      } catch(e){
        btn.textContent = 'Failed';
        setTimeout(()=>btn.textContent='Copy', 1200);
      }
    });
    pre.style.position = 'relative';
    btn.style.position = 'absolute';
    btn.style.top = '8px';
    btn.style.right = '8px';
    pre.appendChild(btn);
  });

  // Upgrade any SVG code blocks into inline images
  upgradeSvgBlocks(wrapper);
  // If there are markdown images with empty src, try to backfill from inline SVG in the text
  fixMarkdownImages(wrapper, md || '');
  // If no <img> tags exist but we can detect a data URI or inline SVG, append an image
  if(!wrapper.querySelector('img')){
    const dataUri = extractFirstSvgDataUri(md || '');
    if(dataUri){
      const img = new Image();
      img.src = dataUri;
      img.alt = 'SVG Image';
      img.className = 'svg-image';
      wrapper.appendChild(img);
    } else {
      const svg = extractFirstSvg(md || '');
      if(svg){
        const img = createSvgImg(svg);
        wrapper.appendChild(img);
      }
    }
  }
  return wrapper;
}

function addMessage(role, text){
  const el = document.createElement('div');
  el.className = `message ${role}`;
  if(role === 'assistant'){
    const svgEl = renderSvgIfAny(text);
    if(svgEl){
      el.appendChild(svgEl);
    } else {
      const content = renderMarkdownToHtml(text);
      el.appendChild(content);
    }
  }else{
    el.textContent = text;
  }
  $('#messages').appendChild(el);
  el.scrollIntoView({behavior:'smooth', block:'end'});
  history.push({role, content: text});
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

function setTyping(show){
  const t = $('#typing');
  t.hidden = !show;
  $('#sendBtn').disabled = !!show;
}

async function sendMessage(){
  if(sending) return;
  sending = true;
  const apiBase = API_BASE || '';
  const apiKey = $('#apiKey').value.trim();
  const input = $('#userInput');
  const content = input.value.trim();
  if(!content) return;
  input.value = '';
  addMessage('user', content);
  setTyping(true);

  const res = await fetch(`${apiBase}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': apiKey ? `Bearer ${apiKey}` : ''
    },
    body: JSON.stringify({
      model: DEFAULT_MODEL,
      messages: buildOpenAIMessages(history, content),
      max_tokens: 256
    })
  });

  if(!res.ok){
    const t = await res.text();
    addMessage('assistant', `Error: ${t}`);
    setTyping(false);
    sending = false;
    return;
  }

  const data = await res.json();
  const text = data.choices?.[0]?.message?.content ?? '(no content)';
  addMessage('assistant', text);
  setTyping(false);
  sending = false;
}

$('#sendBtn').addEventListener('click', sendMessage);
$('#userInput').addEventListener('keydown', e => {
  // Shift+Enter => newline. Enter => send.
  if(e.key === 'Enter'){
    if(e.shiftKey){
      return; // allow newline
    }
    e.preventDefault();
    sendMessage();
  }
  // Also support Cmd/Ctrl+Enter as send
  if(e.key === 'Enter' && (e.metaKey || e.ctrlKey)){
    e.preventDefault();
    sendMessage();
  }
});

// history management
function loadHistory(){
  try{
    history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    history.forEach(m => addMessage(m.role, m.content));
  }catch(e){ history = []; }
}

function newChat(){
  history = [];
  localStorage.setItem(STORAGE_KEY, '[]');
  $('#messages').innerHTML = '';
}

function buildOpenAIMessages(hist, latestUserContent){
  const msgs = [];
  // optional: system prompt could be added here
  hist.forEach(m => msgs.push({ role: m.role, content: m.content }));
  msgs.push({ role: 'user', content: latestUserContent });
  return msgs;
}

$('#newChatBtn').addEventListener('click', newChat);

// init
loadHistory();

// ---------------- SVG helpers ----------------
function renderSvgIfAny(md){
  const trimmed = (md || '').trim();
  // Case 1: message is raw SVG
  if(isSvg(trimmed)){
    return createSvgInline(trimmed);
  }
  // Case 1b: message is already a data URI for SVG
  if(/^data:image\/svg\+xml(;charset=[^,]+)?,/i.test(trimmed)){
    // Decode to inline SVG for reliability
    try{
      const content = decodeURIComponent(trimmed.split(',').slice(1).join(','));
      return createSvgInline(content);
    }catch(_){
      return createSvgInline(trimmed);
    }
  }
  // Case 2: message is a single fenced block that contains SVG
  const fenced = trimmed.match(/^```(\w+)?\n([\s\S]*?)\n```\s*$/);
  if(fenced){
    const lang = (fenced[1] || '').toLowerCase();
    const code = fenced[2] || '';
    if((lang === 'svg' || lang === 'xml' || lang === 'html') && isSvg(code.trim())){
      return createSvgInline(code);
    }
    // If no language was specified but content is SVG
    if(!lang && isSvg(code.trim())){
      return createSvgInline(code);
    }
  }
  return null;
}

function isSvg(s){
  if(!s) return false;
  let t = s.trim();
  // strip XML declaration and DOCTYPE
  t = t.replace(/^<\?xml[^>]*>/i, '').trim();
  t = t.replace(/<!DOCTYPE[^>]*>/i, '').trim();
  return t.startsWith('<svg') && t.includes('</svg>');
}

function sanitizeSvg(svg){
  let s = (svg || '').trim();
  // Remove XML declaration and DOCTYPE
  s = s.replace(/^<\?xml[^>]*>/i, '').trim();
  s = s.replace(/<!DOCTYPE[\s\S]*?>/i, '').trim();
  // Basic safety: strip script tags
  s = s.replace(/<script[\s\S]*?<\/script>/gi, '');
  // Ensure xmlns is present on root <svg>
  s = s.replace(/<svg(\s[^>]*?)?>/i, (m)=>{
    if(/xmlns=/.test(m)) return m;
    return m.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
  });
  return s;
}

function createSvgInline(svg){
  const wrapper = document.createElement('div');
  wrapper.className = 'svg-inline-wrapper';
  wrapper.innerHTML = sanitizeSvg(svg);
  const node = wrapper.firstElementChild || wrapper;
  if(node && node.tagName && node.tagName.toLowerCase() === 'svg'){
    node.classList.add('svg-inline');
  }
  return node;
}

function upgradeSvgBlocks(container){
  container.querySelectorAll('pre code').forEach(code => {
    const text = (code.textContent || '').trim();
    const cls = (code.className || '').toLowerCase();
    const looksLikeSvgLang = cls.includes('language-svg') || cls.includes('language-xml') || cls.includes('language-html') || cls.includes('language-svg+xml');
    if((looksLikeSvgLang && isSvg(text)) || isSvg(text)){
      const inline = createSvgInline(text);
      const pre = code.closest('pre');
      if(pre){
        pre.replaceWith(inline);
      }
    }
  });
}

function extractFirstSvg(text){
  const m = (text || '').match(/<svg[\s\S]*?<\/svg>/i);
  return m ? m[0] : null;
}

function extractFirstSvgDataUri(text){
  const m = (text || '').match(/data:image\/svg\+xml(?:;charset=[^,]+)?,[^\s)"']+/i);
  return m ? m[0] : null;
}

function fixMarkdownImages(container, rawMd){
  const imgs = container.querySelectorAll('img');
  if(!imgs.length) return;
  const svg = extractFirstSvg(rawMd);
  const dataUri = extractFirstSvgDataUri(rawMd);
  imgs.forEach(img => {
    const src = img.getAttribute('src') || '';
    if(!src || src === '#'){
      let node = null;
      if(dataUri){
        try{
          const content = decodeURIComponent(dataUri.split(',').slice(1).join(','));
          node = createSvgInline(content);
        }catch(_){
          // fallback inline wrapper containing the URI text
          node = createSvgInline(dataUri);
        }
      } else if(svg){
        node = createSvgInline(svg);
      }
      if(node){
        img.replaceWith(node);
      }
    }
  });
}
