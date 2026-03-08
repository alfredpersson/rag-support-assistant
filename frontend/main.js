'use strict';

const fab       = document.getElementById('chatFab');
const panel     = document.getElementById('chatPanel');
const closeBtn  = document.getElementById('chatClose');
const thread    = document.getElementById('chatThread');
const input     = document.getElementById('chatInput');
const sendBtn   = document.getElementById('chatSend');

let isOpen = false;
let isLoading = false;

// ── Open / close ───────────────────────────────────────────────
function openPanel() {
  isOpen = true;
  panel.classList.add('open');
  panel.setAttribute('aria-hidden', 'false');
  fab.classList.add('open');
  input.focus();
}

function closePanel() {
  isOpen = false;
  panel.classList.remove('open');
  panel.setAttribute('aria-hidden', 'true');
  fab.classList.remove('open');
}

fab.addEventListener('click', () => (isOpen ? closePanel() : openPanel()));
closeBtn.addEventListener('click', closePanel);

// Close on Escape
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && isOpen) closePanel();
});

// ── Welcome message ────────────────────────────────────────────
const SUGGESTIONS = [
  'How do I connect a custom domain?',
  'How do I add a product to my online store?',
  'Can I change my Wix plan?',
];

function renderWelcome() {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg--assistant';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = "Hi! I'm your Wix support assistant. Ask me anything about your site, domains, billing, or any Wix feature.";
  wrap.appendChild(bubble);

  const suggs = document.createElement('div');
  suggs.className = 'suggestions';
  SUGGESTIONS.forEach((q) => {
    const btn = document.createElement('button');
    btn.className = 'suggestion-btn';
    btn.textContent = q;
    btn.addEventListener('click', () => {
      if (isLoading) return;
      input.value = q;
      submitQuestion(q);
    });
    suggs.appendChild(btn);
  });
  wrap.appendChild(suggs);
  thread.appendChild(wrap);
}

// ── Message rendering ──────────────────────────────────────────
function appendMessage(role, text, sources = []) {
  const wrap = document.createElement('div');
  wrap.className = `msg msg--${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);

  if (role === 'assistant' && sources.length) {
    const sourceRow = document.createElement('div');
    sourceRow.className = 'sources';
    sources.forEach((src) => {
      const pill = document.createElement('span');
      pill.className = 'source-pill';
      pill.textContent = src;
      pill.title = src;
      sourceRow.appendChild(pill);
    });
    wrap.appendChild(sourceRow);
  }

  thread.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function appendError(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg--assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble bubble--error';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  thread.appendChild(wrap);
  scrollToBottom();
}

function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg--assistant';
  wrap.id = 'typingIndicator';

  const indicator = document.createElement('div');
  indicator.className = 'typing-indicator';
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement('div');
    dot.className = 'typing-dot';
    indicator.appendChild(dot);
  }
  wrap.appendChild(indicator);
  thread.appendChild(wrap);
  scrollToBottom();
}

function hideTyping() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

function scrollToBottom() {
  thread.scrollTop = thread.scrollHeight;
}

// ── API call ───────────────────────────────────────────────────
async function submitQuestion(question) {
  const q = question.trim();
  if (!q || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;
  input.value = '';
  autoResize();

  appendMessage('user', q);
  showTyping();

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });

    hideTyping();

    if (res.status === 429) {
      appendError("You've reached the daily limit of 100 questions. Please try again tomorrow.");
      return;
    }

    if (!res.ok) {
      appendError('Something went wrong on our end. Please try again in a moment.');
      return;
    }

    const data = await res.json();
    appendMessage('assistant', data.answer, data.sources ?? []);
  } catch {
    hideTyping();
    appendError('Unable to reach the support service. Please check your connection.');
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

// ── Input handling ─────────────────────────────────────────────
function autoResize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
}

input.addEventListener('input', autoResize);

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitQuestion(input.value);
  }
});

sendBtn.addEventListener('click', () => submitQuestion(input.value));

// ── Init ───────────────────────────────────────────────────────
renderWelcome();
