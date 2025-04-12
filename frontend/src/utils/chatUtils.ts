export const generateChatTitle = (messages: Array<{ content: string }>) => {
  if (messages.length === 0) return 'New Chat';
  const firstMessage = messages[0].content;
  return firstMessage.slice(0, 30) + (firstMessage.length > 30 ? '...' : '');
};

export const handleAIResponse = async (message: string): Promise<string> => {
  // Simulated AI response for local development
  await new Promise(resolve => setTimeout(resolve, 1000));
  return `This is a simulated response to: "${message}"`;
};
