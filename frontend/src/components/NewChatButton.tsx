import React from 'react';

interface NewChatButtonProps {
  onClick: () => void;
}

export const NewChatButton: React.FC<NewChatButtonProps> = ({ onClick }) => {
  try {
    return (
      <button
        onClick={onClick}
        className="new-chat-button"
        data-name="new-chat-button"
      >
        <i className="fas fa-plus"></i>
        <span>New Chat</span>
      </button>
    );
  } catch (error) {
    console.error('NewChatButton error:', error);
    return null;
  }
}
