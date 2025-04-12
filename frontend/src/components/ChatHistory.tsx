import { useEffect, useRef } from "react";
import { ChatMessage } from "./ChatMessage";
import { Message } from "../types";

interface ChatHistoryProps {
  messages: Message[];
}

const ChatHistory: React.FC<ChatHistoryProps> = ({ messages }) => {
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-container" data-name="chat-history">
      {messages.map((message, index) => (
        <ChatMessage key={index} message={message} />
      ))}
      <div ref={chatEndRef} />
    </div>
  );
};

export default ChatHistory;
