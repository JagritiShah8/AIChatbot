/**
 * AI Chatbot — Client-side Application Logic
 *
 * Implements multi-conversation state management, localStorage persistence,
 * real-time API communication with Flask backend (/api/chat), safe HTML rendering,
 * and responsive UI interactions.
 */

(() => {
  'use strict';

  // Storage key for persisting conversations across browser refreshes
  const STORAGE_KEY = 'ai_chatbot_conversations_v2';

  // State object
  const state = {
    conversations: [],
    activeConversationId: null,
    isGenerating: false,
  };

  // DOM Elements
  const elements = {
    sidebar: document.getElementById('sidebar'),
    sidebarBackdrop: document.getElementById('sidebarBackdrop'),
    sidebarCloseBtn: document.getElementById('sidebarCloseBtn'),
    mobileMenuBtn: document.getElementById('mobileMenuBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    conversationsList: document.getElementById('conversationsList'),
    currentChatTitle: document.getElementById('currentChatTitle'),
    clearChatBtn: document.getElementById('clearChatBtn'),
    infoModalBtn: document.getElementById('infoModalBtn'),
    aboutModal: document.getElementById('aboutModal'),
    modalCloseBtn: document.getElementById('modalCloseBtn'),
    modalConfirmBtn: document.getElementById('modalConfirmBtn'),
    chatViewport: document.getElementById('chatViewport'),
    welcomeScreen: document.getElementById('welcomeScreen'),
    messagesList: document.getElementById('messagesList'),
    typingIndicator: document.getElementById('typingIndicator'),
    chatForm: document.getElementById('chatForm'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
  };

  // ==========================================================================
  // Storage & State Initialization
  // ==========================================================================

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed.conversations) && parsed.conversations.length > 0) {
          state.conversations = parsed.conversations;
          state.activeConversationId = parsed.activeConversationId || state.conversations[0].id;
          // Ensure active conversation exists
          const exists = state.conversations.some(c => c.id === state.activeConversationId);
          if (!exists) {
            state.activeConversationId = state.conversations[0].id;
          }
          return;
        }
      }
    } catch (err) {
      console.warn('Failed to load conversation history from localStorage:', err);
    }

    // Default state: create initial empty conversation
    createNewConversation(false);
  }

  function saveState() {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          activeConversationId: state.activeConversationId,
          conversations: state.conversations,
        })
      );
    } catch (err) {
      console.warn('Failed to save conversations to localStorage:', err);
    }
  }

  function getActiveConversation() {
    return state.conversations.find(c => c.id === state.activeConversationId) || null;
  }

  function generateId(prefix = 'conv') {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  }

  function formatTime(date = new Date()) {
    return new Intl.DateTimeFormat('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    }).format(date);
  }

  // ==========================================================================
  // Conversation Management
  // ==========================================================================

  function createNewConversation(shouldSave = true) {
    const newConv = {
      id: generateId('conv'),
      title: 'New Conversation',
      createdAt: Date.now(),
      messages: [],
    };

    state.conversations.unshift(newConv);
    state.activeConversationId = newConv.id;

    if (shouldSave) {
      saveState();
    }

    renderConversationsList();
    renderCurrentConversation();
    focusInput();
    closeMobileSidebar();
  }

  function switchConversation(convId) {
    if (state.isGenerating) return;
    const target = state.conversations.find(c => c.id === convId);
    if (!target) return;

    state.activeConversationId = convId;
    saveState();
    renderConversationsList();
    renderCurrentConversation();
    focusInput();
    closeMobileSidebar();
  }

  function deleteConversation(convId, event) {
    if (event) {
      event.stopPropagation();
    }

    const index = state.conversations.findIndex(c => c.id === convId);
    if (index === -1) return;

    state.conversations.splice(index, 1);

    if (state.conversations.length === 0) {
      createNewConversation(true);
      return;
    }

    if (state.activeConversationId === convId) {
      // Pick next adjacent conversation
      const nextIndex = Math.min(index, state.conversations.length - 1);
      state.activeConversationId = state.conversations[nextIndex].id;
    }

    saveState();
    renderConversationsList();
    renderCurrentConversation();
  }

  function clearCurrentChat() {
    const active = getActiveConversation();
    if (!active || active.messages.length === 0) return;

    if (confirm('Are you sure you want to clear messages in this conversation?')) {
      active.messages = [];
      active.title = 'New Conversation';
      saveState();
      renderConversationsList();
      renderCurrentConversation();
    }
  }

  // ==========================================================================
  // Safe HTML & Markdown Formatter (XSS Protection)
  // ==========================================================================

  function escapeHTML(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatBotMarkdown(rawText) {
    if (!rawText) return '';

    // Step 1: Escape raw HTML to prevent injection
    let text = escapeHTML(rawText);

    // Step 2: Code blocks ```python\ncode\n```
    text = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
      return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Step 3: Inline code `code`
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Step 4: Bold **text**
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Step 5: Italic *text*
    text = text.replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>');

    // Step 6: Bullet lists (- item or * item)
    const lines = text.split('\n');
    const resultLines = [];
    let inList = false;

    for (let line of lines) {
      const trimmed = line.trim();
      const listMatch = trimmed.match(/^[-*]\s+(.+)$/);

      if (listMatch) {
        if (!inList) {
          resultLines.push('<ul>');
          inList = true;
        }
        resultLines.push(`<li>${listMatch[1]}</li>`);
      } else {
        if (inList) {
          resultLines.push('</ul>');
          inList = false;
        }
        resultLines.push(line);
      }
    }

    if (inList) {
      resultLines.push('</ul>');
    }

    text = resultLines.join('\n');

    // Step 7: Paragraphs for remaining text blocks
    return text
      .split(/\n{2,}/)
      .map(block => {
        const trimmed = block.trim();
        if (!trimmed) return '';
        if (
          trimmed.startsWith('<pre>') ||
          trimmed.startsWith('<ul>') ||
          trimmed.startsWith('<ol>')
        ) {
          return trimmed;
        }
        return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
      })
      .join('');
  }

  // ==========================================================================
  // Rendering UI
  // ==========================================================================

  function renderConversationsList() {
    elements.conversationsList.innerHTML = '';

    if (state.conversations.length === 0) {
      const hint = document.createElement('div');
      hint.className = 'empty-conversations-hint';
      hint.textContent = 'No conversations yet';
      elements.conversationsList.appendChild(hint);
      return;
    }

    state.conversations.forEach(conv => {
      const item = document.createElement('div');
      item.className = `nav-item ${conv.id === state.activeConversationId ? 'active' : ''}`;
      item.setAttribute('role', 'button');
      item.setAttribute('tabindex', '0');
      item.setAttribute('aria-label', `Conversation: ${conv.title}`);

      // Left icon and title
      const content = document.createElement('div');
      content.className = 'nav-item-content';

      const icon = document.createElement('span');
      icon.className = 'nav-item-icon';
      icon.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>`;

      const title = document.createElement('span');
      title.className = 'nav-item-title';
      title.textContent = conv.title || 'New Conversation';

      content.appendChild(icon);
      content.appendChild(title);
      item.appendChild(content);

      // Delete button
      const delBtn = document.createElement('button');
      delBtn.className = 'btn-delete-conv';
      delBtn.setAttribute('title', 'Delete conversation');
      delBtn.setAttribute('aria-label', `Delete ${conv.title}`);
      delBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;

      delBtn.addEventListener('click', e => deleteConversation(conv.id, e));
      item.appendChild(delBtn);

      item.addEventListener('click', () => switchConversation(conv.id));
      item.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          switchConversation(conv.id);
        }
      });

      elements.conversationsList.appendChild(item);
    });
  }

  function renderCurrentConversation() {
    const active = getActiveConversation();
    if (!active) return;

    elements.currentChatTitle.textContent = active.title || 'New Conversation';
    elements.messagesList.innerHTML = '';

    if (!active.messages || active.messages.length === 0) {
      elements.welcomeScreen.style.display = 'block';
      elements.clearChatBtn.style.opacity = '0.5';
      elements.clearChatBtn.setAttribute('disabled', 'true');
    } else {
      elements.welcomeScreen.style.display = 'none';
      elements.clearChatBtn.style.opacity = '1';
      elements.clearChatBtn.removeAttribute('disabled');

      active.messages.forEach(msg => {
        appendMessageElement(msg, false);
      });
      scrollToBottom(false);
    }
  }

  function appendMessageElement(msg, animate = true) {
    const isUser = msg.role === 'user';
    const row = document.createElement('div');
    row.className = `message-row ${isUser ? 'user' : 'assistant'}`;
    if (!animate) {
      row.style.animation = 'none';
    }

    const avatar = document.createElement('div');
    avatar.className = `avatar ${isUser ? 'user-avatar' : 'assistant-avatar'}`;
    avatar.textContent = isUser ? '👤' : '🤖';

    const contentWrap = document.createElement('div');
    contentWrap.className = 'message-content-wrap';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (isUser) {
      bubble.textContent = msg.content;
    } else {
      bubble.innerHTML = formatBotMarkdown(msg.content);
    }

    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = msg.timestamp || formatTime();

    contentWrap.appendChild(bubble);
    contentWrap.appendChild(meta);

    row.appendChild(avatar);
    row.appendChild(contentWrap);

    elements.messagesList.appendChild(row);
  }

  function scrollToBottom(smooth = true) {
    elements.chatViewport.scrollTo({
      top: elements.chatViewport.scrollHeight,
      behavior: smooth ? 'smooth' : 'auto',
    });
  }

  function showTypingIndicator(show = true) {
    elements.typingIndicator.style.display = show ? 'block' : 'none';
    if (show) {
      scrollToBottom(true);
    }
  }

  function focusInput() {
    elements.messageInput.focus();
  }

  function updateSendButtonState() {
    const text = elements.messageInput.value.trim();
    elements.sendBtn.disabled = text.length === 0 || state.isGenerating;
  }

  function autoResizeInput() {
    const el = elements.messageInput;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
    updateSendButtonState();
  }

  // ==========================================================================
  // Message Sending & API Integration
  // ==========================================================================

  async function handleSendMessage(customText = null) {
    const rawInput = customText !== null ? customText : elements.messageInput.value;
    const cleanedText = rawInput.trim();

    if (!cleanedText || state.isGenerating) {
      return;
    }

    const active = getActiveConversation();
    if (!active) {
      createNewConversation(true);
    }

    const conversation = getActiveConversation();

    // If this is the first message in the conversation, create a friendly title
    if (conversation.messages.length === 0) {
      const generatedTitle =
        cleanedText.length > 28 ? `${cleanedText.slice(0, 28)}...` : cleanedText;
      conversation.title = generatedTitle;
      elements.currentChatTitle.textContent = generatedTitle;
      renderConversationsList();
    }

    // Hide welcome screen
    elements.welcomeScreen.style.display = 'none';
    elements.clearChatBtn.style.opacity = '1';
    elements.clearChatBtn.removeAttribute('disabled');

    // Create user message object
    const userMessage = {
      id: generateId('msg'),
      role: 'user',
      content: cleanedText,
      timestamp: formatTime(),
    };

    conversation.messages.push(userMessage);
    appendMessageElement(userMessage, true);
    saveState();

    // Reset input
    elements.messageInput.value = '';
    autoResizeInput();
    scrollToBottom(true);

    // Lock sending and show typing indicator
    state.isGenerating = true;
    updateSendButtonState();
    showTypingIndicator(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: cleanedText }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || `HTTP error ${response.status}`);
      }

      // Create assistant message
      const botMessage = {
        id: generateId('msg'),
        role: 'assistant',
        content: data.response,
        intent: data.intent,
        timestamp: formatTime(),
      };

      conversation.messages.push(botMessage);
      saveState();

      showTypingIndicator(false);
      appendMessageElement(botMessage, true);
      scrollToBottom(true);
    } catch (err) {
      console.error('Error contacting chat API:', err);
      showTypingIndicator(false);

      const errorMessage = {
        id: generateId('msg'),
        role: 'assistant',
        content:
          '⚠️ Sorry, I encountered an issue connecting to the response engine. Please make sure the server is running and try again.',
        timestamp: formatTime(),
      };

      conversation.messages.push(errorMessage);
      saveState();
      appendMessageElement(errorMessage, true);
      scrollToBottom(true);
    } finally {
      state.isGenerating = false;
      updateSendButtonState();
      focusInput();
    }
  }

  // ==========================================================================
  // Mobile Sidebar & Modal Controls
  // ==========================================================================

  function openMobileSidebar() {
    elements.sidebar.classList.add('open');
    elements.sidebarBackdrop.classList.add('active');
  }

  function closeMobileSidebar() {
    elements.sidebar.classList.remove('open');
    elements.sidebarBackdrop.classList.remove('active');
  }

  function openModal() {
    elements.aboutModal.style.display = 'flex';
  }

  function closeModal() {
    elements.aboutModal.style.display = 'none';
  }

  // ==========================================================================
  // Event Listeners
  // ==========================================================================

  function attachEventListeners() {
    // Form submit
    elements.chatForm.addEventListener('submit', e => {
      e.preventDefault();
      handleSendMessage();
    });

    // Textarea input & keyboard shortcuts
    elements.messageInput.addEventListener('input', autoResizeInput);
    elements.messageInput.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    // New Chat Button
    elements.newChatBtn.addEventListener('click', () => createNewConversation(true));

    // Clear Chat Button
    elements.clearChatBtn.addEventListener('click', clearCurrentChat);

    // Suggested prompt cards
    document.querySelectorAll('.prompt-card').forEach(card => {
      card.addEventListener('click', () => {
        const promptText = card.getAttribute('data-prompt');
        if (promptText) {
          handleSendMessage(promptText);
        }
      });
    });

    // Mobile sidebar
    elements.mobileMenuBtn.addEventListener('click', openMobileSidebar);
    elements.sidebarCloseBtn.addEventListener('click', closeMobileSidebar);
    elements.sidebarBackdrop.addEventListener('click', closeMobileSidebar);

    // About modal
    elements.infoModalBtn.addEventListener('click', openModal);
    elements.modalCloseBtn.addEventListener('click', closeModal);
    elements.modalConfirmBtn.addEventListener('click', closeModal);
    elements.aboutModal.addEventListener('click', e => {
      if (e.target === elements.aboutModal) {
        closeModal();
      }
    });

    // Global keyboard shortcuts
    window.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        closeModal();
        closeMobileSidebar();
      }
    });
  }

  // ==========================================================================
  // App Bootstrapping
  // ==========================================================================

  function init() {
    loadState();
    attachEventListeners();
    autoResizeInput();
    focusInput();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

