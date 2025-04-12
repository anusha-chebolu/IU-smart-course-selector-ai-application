export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface Chat {
  id: number;
  title: string;
  messages: Message[];
}
