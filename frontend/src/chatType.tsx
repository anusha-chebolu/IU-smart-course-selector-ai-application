// Chat interface
export interface Chat {
  id: string;
  userId: string;
  title: string;
  created_at: string;
}

// Chat message interface
export interface ChatMessage {
  id: string;
  chatId: string;
  role: "user" | "assistant";
  message: string | object;
  created_at: string;
}
