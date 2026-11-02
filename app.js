const API_KEY = 'jawad-assistant-secret-2026';

const messagesEl = document.getElementById('chat-messages');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const nameEl = document.querySelector('.chat-header h1');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// NEW: which shop this page is for, read from the URL, e.g.  /?shop=bangash
// If there is no ?shop= the server uses its default shop.
const SHOP = new URLSearchParams(window.location.search).get('shop') || '';

function getOrCreateId(key) {
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}
const USER_ID = getOrCreateId('khyber_user_id');

// NEW: each shop keeps its own conversation, so chats never mix between shops.
const CONV_KEY = 'khyber_conversation_id' + (SHOP ? ':' + SHOP : '');
let conversationId = localStorage.getItem(CONV_KEY) || null;

function setConversationId(id) {
  if (id && id !== conversationId) {
    conversationId = id;
    localStorage.setItem(CONV_KEY, id);
  }
}

function addBubble(text, sender) {
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + sender;
  bubble.textContent = text;
  messagesEl.appendChild(bubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addVideoBubble(videoUrl) {
  const bubble = document.createElement('div');
  bubble.className = 'bubble bot';
  bubble.style.padding = '6px';
  const video = document.createElement('video');
  video.src = videoUrl;
  video.controls = true;
  video.style.maxWidth = '100%';
  video.style.borderRadius = '10px';
  bubble.appendChild(video);
  messagesEl.appendChild(bubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Renders a product picture in the chat, same treatment as video.
function addImageBubble(imageUrl) {
  const bubble = document.createElement('div');
  bubble.className = 'bubble bot';
  bubble.style.padding = '6px';
  const img = document.createElement('img');
  img.src = imageUrl;
  img.alt = 'Product photo';
  img.style.maxWidth = '100%';
  img.style.display = 'block';
  img.style.borderRadius = '10px';
  img.onerror = () => { bubble.remove(); }; // fail quietly if the image can't load
  bubble.appendChild(img);
  messagesEl.appendChild(bubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Shows one picture for every product in the reply (image_urls), or the
// single image_url if the server only sent one.
function showImages(data) {
  const urls = data.image_urls && data.image_urls.length
    ? data.image_urls
    : (data.image_url ? [data.image_url] : []);
  for (const url of urls) {
    addImageBubble(url);
  }
}

function showTyping() {
  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.id = 'typing-indicator';
  typing.innerHTML = '<span></span><span></span><span></span>';
  messagesEl.appendChild(typing);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideTyping() {
  const typing = document.getElementById('typing-indicator');
  if (typing) typing.remove();
}

let GREETING_TEXT = "Hey! I'm Jawad's assistant. What can I help with?";

async function loadSettings() {
  try {
    const response = await fetch('/settings');
    const data = await response.json();
    nameEl.textContent = data.assistant_name;
    GREETING_TEXT = data.greeting;
  } catch (error) {
    console.error('Could not load settings:', error);
  }
  await restoreConversation();
}

async function restoreConversation() {
  if (!conversationId) {
    messagesEl.innerHTML = '';
    addBubble(GREETING_TEXT, 'bot');
    return;
  }
  try {
    const response = await fetch(`/conversations/${conversationId}/messages?user_id=${USER_ID}`);
    if (!response.ok) throw new Error('history fetch failed');
    const messages = await response.json();
    if (!messages || messages.length === 0) {
      messagesEl.innerHTML = '';
      addBubble(GREETING_TEXT, 'bot');
      return;
    }
    messagesEl.innerHTML = '';
    for (const m of messages) {
      if (m.message_type === 'image') {
        addImageBubble(m.message);
      } else if (m.message_type === 'video') {
        addVideoBubble(m.message);
      } else {
        addBubble(m.message, m.sender === 'user' ? 'user' : 'bot');
      }
    }
  } catch (error) {
    console.error('Could not load conversation history:', error);
    messagesEl.innerHTML = '';
    addBubble(GREETING_TEXT, 'bot');
  }
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  addBubble(text, 'user');
  inputEl.value = '';
  showTyping();

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY
      },
      // NEW: shop is sent with every message (empty = server default shop)
      body: JSON.stringify({
        message: text,
        user_id: USER_ID,
        conversation_id: conversationId,
        shop: SHOP || undefined
      })
    });
    const data = await response.json();
    hideTyping();
    setConversationId(data.conversation_id);
    if (data.handed_off) {
      // A manager has taken this conversation over — the AI stays quiet,
      // so we don't show anything here (the human will type separately,
      // e.g. via the admin panel or another channel).
      return;
    }
    addBubble(data.reply || data.detail || "Sorry, something went wrong.", 'bot');
    showImages(data);
    if (data.video_url) {
      addVideoBubble(data.video_url);
    }
  } catch (error) {
    hideTyping();
    addBubble("Sorry, I couldn't connect right now.", 'bot');
  }
}

sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendMessage();
});

micBtn.addEventListener('click', async () => {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        stream.getTracks().forEach(track => track.stop());
        await sendVoiceMessage(audioBlob);
      };

      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add('recording');
      micBtn.textContent = '⏹';
    } catch (error) {
      console.error('Microphone access denied:', error);
      addBubble("I need microphone access to hear you.", 'bot');
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    micBtn.classList.remove('recording');
    micBtn.textContent = '🎤';
  }
});

// NEW: plays a base64 audio clip. mimeType is audio/mpeg (Edge-TTS)
// or audio/wav (Gemini backup voice).
function playBase64Audio(base64, mimeType) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: mimeType || 'audio/mpeg' });
  const audio = new Audio(URL.createObjectURL(blob));
  // Some browsers block autoplay; ignore that error quietly.
  audio.play().catch((err) => console.warn('Audio play blocked:', err));
}

// NEW: the text reply is already on screen. Now ask the server for the voice
// version of it (one request, nothing is stored between requests).
async function speakReply(text) {
  try {
    const response = await fetch('/tts', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY
      },
      body: JSON.stringify({ text })
    });
    if (!response.ok) return; // text reply is already shown
    const data = await response.json();
    if (data.audio_base64) {
      playBase64Audio(data.audio_base64, data.mime_type);
    }
  } catch (error) {
    console.warn('Voice reply failed:', error);
  }
}

async function sendVoiceMessage(audioBlob) {
  showTyping();
  const reader = new FileReader();
  reader.readAsDataURL(audioBlob);
  reader.onloadend = async () => {
    const base64Audio = reader.result.split(',')[1];
    try {
      const response = await fetch('/voice-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY
        },
        body: JSON.stringify({
          audio_base64: base64Audio,
          mime_type: 'audio/webm',
          user_id: USER_ID,
          conversation_id: conversationId,
          shop: SHOP || undefined
        })
      });
      const data = await response.json();
      hideTyping();
      setConversationId(data.conversation_id);
      if (data.transcript) addBubble(data.transcript, 'user');
      if (data.handed_off) return;

      // 1) TEXT first, immediately
      addBubble(data.reply || data.detail || "Sorry, something went wrong.", 'bot');
      showImages(data);
      if (data.video_url) {
        addVideoBubble(data.video_url);
      }

      // 2) VOICE right after (not awaited, so nothing is blocked)
      if (data.reply) {
        speakReply(data.reply);
      }

    } catch (error) {
      hideTyping();
      addBubble("Sorry, I couldn't process that voice message.", 'bot');
    }
  };
}

// Lets the customer explicitly start a new conversation, per spec:
// memory/history should only ever be cleared on purpose, never automatically.
document.getElementById('new-chat-btn').addEventListener('click', async () => {
  try {
    const response = await fetch(`/conversations?user_id=${USER_ID}`, { method: 'POST' });
    const data = await response.json();
    setConversationId(data.conversation_id);
    messagesEl.innerHTML = '';
    addBubble(GREETING_TEXT, 'bot');
  } catch (error) {
    console.error('Could not start a new conversation:', error);
  }
});

// Demo mode: quick-tap example messages that show off multilingual
// support, product search, video, ordering, and human handoff at a glance.
document.querySelectorAll('.demo-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    inputEl.value = chip.dataset.msg;
    sendMessage();
  });
});

loadSettings();