import React from 'react';
import { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  try {
    const { role, content } = message;
    const isUser = role === 'user';

    return (
      <div 
        className={`chat-message ${isUser ? 'user-message' : 'assistant-message'}`}
        data-name={`${role}-message`}
      >
        <div className="message-avatar" data-name="message-avatar">
          <i className={`fas ${isUser ? 'fa-user' : 'fa-robot'}`}></i>
        </div>
        <div className="message-content markdown-content" data-name="message-content">
          {content}
        </div>
      </div>
    );
  } catch (error) {
    console.error('ChatMessage error:', error);
    return null;
  }
}
