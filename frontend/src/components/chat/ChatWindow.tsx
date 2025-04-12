import React, { useState, useEffect, useRef } from "react";
import { formatDistanceToNow } from "date-fns";
import {
  useGetChatDetailsQuery,
  useGetChatMessagesQuery,
  useAddChatMessageMutation,
} from "../../services/chat_api";

import { Chat, ChatMessage } from "../../chatType";
import { User } from "../../types";

interface ChatWindowProps {
  chatId: string;
  user: User;
}

// Helper to extract message content safely
const getMessageContent = (message: string | object): string => {
  if (typeof message === "string") {
    return message;
  } else if (message && typeof message === "object") {
    // Check if message has a response property
    if ("response" in message) {
      return (message as any).response;
    }
    // Otherwise stringify the object
    return JSON.stringify(message);
  }
  return "No message content";
};

// Check if the message contains course recommendations
const hasRecommendations = (message: string | object): boolean => {
  if (typeof message === "string") {
    return message.includes("recommended_courses");
  } else if (message && typeof message === "object") {
    return (
      "recommended_courses" in message ||
      ("json_response" in message &&
        (message as any).json_response?.recommended_courses)
    );
  }
  return false;
};

// Extract course recommendations if present
const getRecommendations = (message: string | object): any[] => {
  try {
    if (typeof message === "string") {
      if (message.includes("recommended_courses")) {
        const parsed = JSON.parse(message);
        return (
          parsed.recommended_courses ||
          parsed.json_response?.recommended_courses ||
          []
        );
      }
    } else if (message && typeof message === "object") {
      if ("recommended_courses" in message) {
        return (message as any).recommended_courses || [];
      } else if (
        "json_response" in message &&
        (message as any).json_response?.recommended_courses
      ) {
        return (message as any).json_response.recommended_courses || [];
      }
    }
  } catch (error) {
    console.error("Error parsing recommendations:", error);
  }

  return [];
};

const ChatWindow: React.FC<ChatWindowProps> = ({ chatId, user }) => {
  const [messageText, setMessageText] = useState("");
  const [isSending, setIsSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // RTK Query hooks
  const { data: chat, isLoading: isLoadingChat } =
    useGetChatDetailsQuery(chatId);

  const { data: messages, isLoading: isLoadingMessages } =
    useGetChatMessagesQuery(chatId);
  const [addChatMessage] = useAddChatMessageMutation();

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!messageText.trim() || isSending) return;

    setIsSending(true);

    try {
      // Send user message
      await addChatMessage({
        chatId,
        message: messageText.trim(),
        role: "user",
      }).unwrap();

      setMessageText("");

      // Simulate assistant response after a short delay
      setTimeout(async () => {
        try {
          // Example of sending an object message with recommendations
          // You can modify this to send string or object responses as needed
          const demoResponse = {
            response: `This is an automated response to: "${messageText.trim()}"`,
            recommended_courses: messageText.toLowerCase().includes("course")
              ? [
                  {
                    course_id: "CS101",
                    course_title: "Introduction to Computer Science",
                    course_description:
                      "An introductory course covering basic concepts.",
                  },
                ]
              : undefined,
          };

          await addChatMessage({
            chatId,
            message: demoResponse,
            role: "assistant",
          }).unwrap();
        } catch (error) {
          console.error("Failed to send assistant message:", error);
        }
      }, 1000);
    } catch (error) {
      console.error("Failed to send message:", error);
    } finally {
      setIsSending(false);
    }
  };

  // Render a course recommendation card
  const renderCourseCard = (course: any) => {
    return (
      <div
        key={course.course_id}
        className="mt-2 p-3 bg-gray-50 border border-gray-200 rounded-md"
      >
        <h4 className="font-medium text-blue-700">
          {course.course_id}: {course.course_title}
        </h4>
        {course.course_description && (
          <p className="text-sm text-gray-700 mt-1">
            {course.course_description}
          </p>
        )}
      </div>
    );
  };

  if (isLoadingChat) {
    return (
      <div className="flex-1 flex items-center justify-center">
        Loading chat...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Chat Header */}
      <div className="p-4 border-b border-gray-200 bg-white">
        <h2 className="text-lg font-semibold">{chat?.title || "Chat"}</h2>
        <p className="text-sm text-gray-500">
          Created{" "}
          {chat
            ? formatDistanceToNow(new Date(chat.created_at), {
                addSuffix: true,
              })
            : ""}
        </p>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
        {isLoadingMessages ? (
          <div className="text-center py-4 text-gray-500">
            Loading messages...
          </div>
        ) : messages && messages.length > 0 ? (
          <div className="space-y-4">
            {messages.map((message) => {
              const content = getMessageContent(message.message);
              const recommendations = hasRecommendations(message.message)
                ? getRecommendations(message.message)
                : [];

              return (
                <div
                  key={message.id}
                  className={`flex ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[70%] p-3 rounded-lg ${
                      message.role === "user"
                        ? "bg-blue-500 text-white rounded-br-none"
                        : "bg-white border border-gray-200 rounded-bl-none"
                    }`}
                  >
                    <div className="text-sm">{content}</div>

                    {/* Render course recommendations if present */}
                    {message.role === "assistant" &&
                      recommendations.length > 0 && (
                        <div className="mt-3 pt-2 border-t border-gray-200">
                          <div className="text-sm font-medium text-gray-800 mb-1">
                            Recommended Courses:
                          </div>
                          {recommendations.map((course) =>
                            renderCourseCard(course)
                          )}
                        </div>
                      )}

                    <div
                      className={`text-xs mt-1 ${
                        message.role === "user"
                          ? "text-blue-100"
                          : "text-gray-500"
                      }`}
                    >
                      {formatDistanceToNow(new Date(message.created_at), {
                        addSuffix: true,
                      })}
                    </div>
                  </div>
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            No messages yet. Start the conversation!
          </div>
        )}
      </div>

      {/* Message Input */}
      <div className="p-4 border-t border-gray-200 bg-white">
        <form onSubmit={handleSendMessage} className="flex items-center">
          <input
            type="text"
            value={messageText}
            onChange={(e) => setMessageText(e.target.value)}
            placeholder="Type a message..."
            className="flex-1 p-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-1 focus:ring-blue-500"
            disabled={isSending}
          />
          <button
            type="submit"
            disabled={isSending || !messageText.trim()}
            className="bg-blue-600 text-white px-4 py-2 rounded-r-md hover:bg-blue-700 focus:outline-none disabled:opacity-50"
          >
            {isSending ? "Sending..." : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatWindow;
