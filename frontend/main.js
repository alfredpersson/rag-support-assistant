'use strict';

const fab       = document.getElementById('chatFab');
const panel     = document.getElementById('chatPanel');
const closeBtn  = document.getElementById('chatClose');
const expandBtn = document.getElementById('chatExpand');
const thread    = document.getElementById('chatThread');
const input     = document.getElementById('chatInput');
const sendBtn   = document.getElementById('chatSend');

let isOpen    = false;
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
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && isOpen) closePanel(); });

// ── Expand / collapse ──────────────────────────────────────────
let isExpanded = false;
expandBtn.addEventListener('click', () => {
  isExpanded = !isExpanded;
  panel.classList.toggle('expanded', isExpanded);
  // Swap icon: show "collapse" arrows when expanded
  expandBtn.querySelector('svg path').setAttribute(
    'd',
    isExpanded
      ? 'M5 1v4H1M9 1h4v4M9 13v-4h4M5 13H1V9'   // compress icon
      : 'M1 5V1h4M9 1h4v4M13 9v4H9M5 13H1V9'    // expand icon
  );
});

// ── Minimal Markdown → HTML ────────────────────────────────────
// Handles: **bold**, numbered lists, alternative-method headings, line breaks
function renderMarkdown(text) {
  const lines = text.split('\n');
  const html  = [];
  let inOl    = false;

  function closeOl() {
    if (inOl) { html.push('</ol>'); inOl = false; }
  }

  function inlineFormat(line) {
    return line
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  }

  let lastWasBlank = false;
  for (let raw of lines) {
    const trimmed = raw.trim();

    // Numbered list item: "1. …" or "1) …"
    const listMatch = trimmed.match(/^(\d+)[.)]\s+(.+)$/);
    if (listMatch) {
      if (!inOl) { html.push('<ol>'); inOl = true; }
      html.push(`<li>${inlineFormat(listMatch[2])}</li>`);
      lastWasBlank = false;
      continue;
    }

    closeOl();

    if (trimmed === '') {
      if (!lastWasBlank) html.push('<br>');
      lastWasBlank = true;
      continue;
    }

    lastWasBlank = false;

    // Headings: ###, ##, #
    const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      const level = Math.min(headingMatch[1].length + 3, 6); // h4/h5/h6 to stay small
      html.push(`<h${level}>${inlineFormat(headingMatch[2])}</h${level}>`);
      continue;
    }

    html.push(`<p>${inlineFormat(trimmed)}</p>`);
  }
  closeOl();
  return html.join('').replace(/<br>(<ol)/g, '$1');
}

// ── Step truncation ────────────────────────────────────────────
const MAX_VISIBLE_STEPS = 4;

// Returns { visibleHtml, hiddenHtml } split at step boundary
function splitSteps(fullHtml) {
  const parser = new DOMParser();
  const doc    = parser.parseFromString(`<div>${fullHtml}</div>`, 'text/html');
  const root   = doc.querySelector('div');
  const nodes  = Array.from(root.childNodes);

  // Collect list items across all <ol> elements
  let stepCount  = 0;
  let splitNode  = null; // the <ol> that gets split
  let splitIndex = -1;   // index within that <ol>

  for (const node of nodes) {
    if (node.nodeName === 'OL') {
      const items = Array.from(node.querySelectorAll('li'));
      if (stepCount + items.length > MAX_VISIBLE_STEPS) {
        splitNode  = node;
        splitIndex = MAX_VISIBLE_STEPS - stepCount;
        break;
      }
      stepCount += items.length;
    }
  }

  if (!splitNode) return { visibleHtml: fullHtml, hiddenHtml: '' };

  // Build visible: everything before splitNode + first splitIndex items of splitNode
  let visibleHtml = '';
  for (const node of nodes) {
    if (node === splitNode) break;
    visibleHtml += node.outerHTML ?? node.textContent;
  }
  const visItems = Array.from(splitNode.querySelectorAll('li'));
  if (splitIndex > 0) {
    visibleHtml += '<ol>' + visItems.slice(0, splitIndex).map(li => li.outerHTML).join('') + '</ol>';
  }

  // Build hidden: remaining items continue the numbered list from the correct position
  const hiddenStart = splitIndex + 1;
  let hiddenHtml = `<ol start="${hiddenStart}">` + visItems.slice(splitIndex).map(li => li.outerHTML).join('') + '</ol>';
  let found = false;
  for (const node of nodes) {
    if (node === splitNode) { found = true; continue; }
    if (found) hiddenHtml += node.outerHTML ?? node.textContent;
  }

  return { visibleHtml, hiddenHtml };
}

// ── Message rendering ──────────────────────────────────────────
function scrollToBottom() { thread.scrollTop = thread.scrollHeight; }

function appendUserMessage(text) {
  const wrap   = document.createElement('div');
  wrap.className = 'msg msg--user';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  thread.appendChild(wrap);
  scrollToBottom();
}

function appendAssistantMessage(text, sources = [], routing = '') {
  const fullHtml = renderMarkdown(text);
  const { visibleHtml, hiddenHtml } = splitSteps(fullHtml);

  const wrap = document.createElement('div');
  wrap.className = 'msg msg--assistant';

  const bubble = document.createElement('div');
  bubble.className = 'bubble bubble--md';
  bubble.innerHTML = visibleHtml;

  if (hiddenHtml) {
    const hidden = document.createElement('div');
    hidden.className = 'steps-overflow hidden';
    hidden.innerHTML = hiddenHtml;
    bubble.appendChild(hidden);

    const showBtn = document.createElement('button');
    showBtn.className = 'show-steps-btn';
    showBtn.textContent = 'Show all steps ▾';
    showBtn.addEventListener('click', () => {
      hidden.classList.toggle('hidden');
      showBtn.textContent = hidden.classList.contains('hidden') ? 'Show all steps ▾' : 'Show less ▴';
      scrollToBottom();
    });
    bubble.appendChild(showBtn);
  }

  wrap.appendChild(bubble);

  if (routing === 'answered' && sources.length) {
    const sourceRow = document.createElement('div');
    sourceRow.className = 'sources';
    const label = document.createElement('span');
    label.className = 'sources-label';
    label.textContent = 'Read more:';
    sourceRow.appendChild(label);
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

function appendBotText(text) {
  const wrap   = document.createElement('div');
  wrap.className = 'msg msg--assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble bubble--md';
  bubble.innerHTML = renderMarkdown(text);
  wrap.appendChild(bubble);
  thread.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function appendError(text) {
  const wrap   = document.createElement('div');
  wrap.className = 'msg msg--assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble bubble--error';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  thread.appendChild(wrap);
  scrollToBottom();
}

function showTyping() {
  const wrap       = document.createElement('div');
  wrap.className   = 'msg msg--assistant';
  wrap.id          = 'typingIndicator';
  const indicator  = document.createElement('div');
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

// ── Routing buttons helper ─────────────────────────────────────
function appendRoutingButtons(options, onSelect) {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg--assistant';
  const btnRow = document.createElement('div');
  btnRow.className = 'routing-btns';
  options.forEach((label) => {
    const btn = document.createElement('button');
    btn.className = 'routing-btn';
    btn.textContent = label;
    btn.addEventListener('click', () => {
      wrap.remove();
      onSelect(label);
    });
    btnRow.appendChild(btn);
  });
  wrap.appendChild(btnRow);
  thread.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

// ── Mock agent availability ────────────────────────────────────
const TEAMS = {
  'Billing':           { agents: 3, wait: '~2 min' },
  'Technical support': { agents: 1, wait: '~8 min' },
  'Something else':    { agents: 2, wait: '~5 min' },
};

function handleTeamSelection(teamLabel) {
  appendUserMessage(teamLabel);

  const key  = Object.keys(TEAMS).find(k => teamLabel.toLowerCase().includes(k.toLowerCase()))
               || 'Something else';
  const info = TEAMS[key] ?? TEAMS['Something else'];

  const availability = info.agents > 0
    ? `There ${info.agents === 1 ? 'is' : 'are'} **${info.agents} agent${info.agents !== 1 ? 's' : ''}** available on the **${key}** team with an estimated wait of **${info.wait}**.`
    : `No agents are currently available on the **${key}** team.`;

  appendBotText(`${availability}\n\nWhen you're ready, a support specialist will join this chat. In the meantime, feel free to keep asking me questions!`);
}

// ── "Talk to a person" flow ────────────────────────────────────
function startHumanHandoff() {
  appendBotText('Sure! To connect you with the right team — **is this about billing, technical support, or something else?**');
  appendRoutingButtons(['Billing', 'Technical support', 'Something else'], handleTeamSelection);
}

// ── Welcome message ────────────────────────────────────────────
const SUGGESTIONS = [
  'How do I connect a custom domain?',
  'How do I add a product to my online store?',
  'Can I change my Wix plan?',
];

function renderWelcome() {
  const wrap   = document.createElement('div');
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
    btn.addEventListener('click', () => { if (!isLoading) submitQuestion(q); });
    suggs.appendChild(btn);
  });
  wrap.appendChild(suggs);
  thread.appendChild(wrap);
}

// ── Connect-to-agent inline button ────────────────────────────
function appendConnectAgentButton() {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg--assistant';
  const btn = document.createElement('button');
  btn.className = 'routing-btn connect-agent-btn';
  btn.textContent = '👤 Connect me to a support agent';
  btn.addEventListener('click', () => {
    wrap.remove();
    startHumanHandoff();
  });
  const row = document.createElement('div');
  row.className = 'routing-btns';
  row.appendChild(btn);
  wrap.appendChild(row);
  thread.appendChild(wrap);
  scrollToBottom();
}

// ── API call ───────────────────────────────────────────────────
async function submitQuestion(question) {
  const q = question.trim();
  if (!q || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;
  input.value = '';
  autoResize();

  appendUserMessage(q);
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
    appendAssistantMessage(data.answer, data.sources ?? [], data.routing ?? '');

    if (['out_of_scope', 'high_stakes', 'cannot_answer'].includes(data.routing)) {
      appendConnectAgentButton();
    }
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
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitQuestion(input.value); }
});
sendBtn.addEventListener('click', () => submitQuestion(input.value));

// ── "Talk to a person" button ──────────────────────────────────
document.getElementById('talkToHuman').addEventListener('click', (e) => {
  e.preventDefault();
  if (!isOpen) openPanel();
  startHumanHandoff();
});

// ── Init ───────────────────────────────────────────────────────
renderWelcome();
