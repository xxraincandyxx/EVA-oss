import React, { useState, useEffect, useRef } from 'react';

import { publicUrl } from '../publicUrl';
import useRobotStore from '../store/useRobotStore';

const Agent = () => {
  const agentMessages = useRobotStore((state) => state.agentMessages);
  const isAgentTyping = useRobotStore((state) => state.isAgentTyping);
  const sendAgentMessage = useRobotStore((state) => state.sendAgentMessage);
  const llmBackend = useRobotStore((state) => state.llmBackend);
  const localLlmUrl = useRobotStore((state) => state.localLlmUrl);
  const setLlmBackend = useRobotStore((state) => state.setLlmBackend);

  const [input, setInput] = useState('');
  const [localUrlInput, setLocalUrlInput] = useState('');
  const chatBoxRef = useRef(null);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [agentMessages, isAgentTyping]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim()) {
      sendAgentMessage(input);
      setInput('');
    }
  };

  return (
    <>
      <div className="header">
        <div className="left">
          <h1>Agent</h1>
          <ul className="breadcrumb">
            <li>
              <a href="#">Dashboard</a>
            </li>
            /
            <li>
              <a href="#" className="active">
                Agent
              </a>
            </li>
          </ul>
        </div>
      </div>
      <div className="bottom-data">
        <div className="orders agent-chat-container">
          <div className="header">
            <i className="bx bx-conversation"></i>
            <h3>Chat with EVA</h3>
          </div>
          <div className="chat-box" ref={chatBoxRef}>
            {agentMessages.length === 0 && !isAgentTyping && (
              <div className="chat-message bot">
                <div className="message-bubble">
                  Hello! I'm EVA, your robotic arm assistant. How can I help you
                  today? You can ask me to move the arm, control peripherals, or
                  check its status.
                </div>
                <div className="avatar">
                  <i className="bx bxs-bot"></i>
                </div>
              </div>
            )}
            {agentMessages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.sender}`}>
                <div
                  className="message-bubble"
                  dangerouslySetInnerHTML={{ __html: msg.text }}
                ></div>
                <div className="avatar">
                  <i
                    className={`bx ${
                      msg.sender === 'bot' ? 'bxs-bot' : 'bxs-user'
                    }`}
                  ></i>
                </div>
              </div>
            ))}
            {isAgentTyping && (
              <div className="chat-message bot typing">
                <div className="message-bubble">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
                <div className="avatar">
                  <i className="bx bxs-bot"></i>
                </div>
              </div>
            )}
          </div>
          <div className="chat-input-box">
            <form id="chat-form" className="chat-form" onSubmit={handleSubmit}>
              <input
                type="text"
                id="chat-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask EVA to do something..."
                autoComplete="off"
              />
              <button type="submit" id="send-btn">
                <i className="bx bxs-send"></i>
              </button>
            </form>
          </div>
        </div>
        <div className="reminders agent-info-panel">
          <div className="header">
            <i className="bx bx-info-circle"></i>
            <h3>Agent Info</h3>
          </div>
          <div className="system-info">
            <img
              src={publicUrl('figures/logo.png')}
              alt="EVA Icon"
              className="eva-icon"
            />
            <p className="system-tagline">AI-Powered Robotic Arm Interface</p>
            <div className="info-grid">
              <div>
                <strong>Language Model</strong>{' '}
                {llmBackend === 'local_llama'
                  ? '本地 Qwen2.5-0.5B（llama.cpp）'
                  : '雲端 Deepseek API'}
              </div>
              <div>
                <strong>Capabilities</strong> Natural Language Control, Status
                Inquiry
              </div>
            </div>
            <div className="llm-switcher" style={{ marginTop: '12px' }}>
              <label style={{ display: 'block', marginBottom: '6px' }}>
                LLM 後端切換
              </label>
              <div
                style={{
                  display: 'flex',
                  gap: '8px',
                  alignItems: 'center',
                  marginBottom: '8px',
                }}
              >
                <button
                  type="button"
                  className={llmBackend === 'local_llama' ? 'active' : ''}
                  onClick={() =>
                    setLlmBackend(
                      'local_llama',
                      localUrlInput || localLlmUrl || ''
                    )
                  }
                >
                  本地 Qwen
                </button>
                <button
                  type="button"
                  className={llmBackend === 'deepseek' ? 'active' : ''}
                  onClick={() => setLlmBackend('deepseek', '')}
                >
                  雲端 Deepseek
                </button>
              </div>
              {llmBackend === 'local_llama' && (
                <input
                  type="text"
                  value={localUrlInput || localLlmUrl || ''}
                  onChange={(e) => setLocalUrlInput(e.target.value)}
                  placeholder="本地 LLM URL，例如 http://127.0.0.1:8080/v1/chat/completions"
                  style={{ width: '100%', fontSize: '12px' }}
                />
              )}
            </div>
            <h4>Example Commands:</h4>
            <ul className="example-commands">
              <li>"Move 10cm forward."</li>
              <li>"Rotate axis 3 by -45 degrees."</li>
              <li>"What is your current position?"</li>
              <li>"打开吸泵 (Turn on the pump)."</li>
            </ul>
          </div>
        </div>
      </div>
    </>
  );
};

export default Agent;
