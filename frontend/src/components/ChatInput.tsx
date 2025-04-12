// components/ChatInput.tsx
import React from "react";

interface ChatInputProps {
  input: string;
  setInput: (input: string) => void;
  isLoading: boolean;
  inputRef: React.RefObject<HTMLInputElement>; // Properly typed as React.RefObject
  onSubmit: (e: React.FormEvent) => void;
}

const ChatInput: React.FC<ChatInputProps> = ({
  input,
  setInput,
  isLoading,
  inputRef,
  onSubmit,
}) => {
  return (
    <form onSubmit={onSubmit} className="input-area">
      <div className="input-container">
        <div className="input-wrapper">
          <input
            ref={inputRef}
            type="text"
            className="message-input"
            placeholder="Describe your career goals or interests..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
          />
          <div className="input-actions">
            <button
              type="button"
              className="action-button"
              title="Upload transcript (optional)"
            >
              <svg
                className="action-icon"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 4v16m8-8H4"
                />
              </svg>
            </button>
            <button
              type="submit"
              className="send-button"
              disabled={!input.trim() || isLoading}
            >
              <svg
                className="send-icon"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M14 5l7 7m0 0l-7 7m7-7H3"
                />
              </svg>
            </button>
          </div>
        </div>
        <div className="input-footer">
          <div className="hint-text">
            Try: "I want to become a data scientist" or "What courses prepare me
            for UX design?"
          </div>
          <div className="model-info">Powered by AI</div>
        </div>
      </div>
    </form>
  );
};

export default ChatInput;
