import React, { useState, useRef, useEffect } from "react";
import { Chat, ChatMessage } from "../../chatType";
import {
  useAddChatMessageMutation,
  useGetChatMessagesQuery,
  useGetChatDetailsQuery,
  useGetAIResponseMutation,
  useGetChatPendingStatusQuery,
  useUpdateChatMutation,
  pendingResponses,
} from "../../services/chat_api";

import { useAddCourseToMyCoursesMutation } from "../../services/mycourses_api"; // Import the new course mutation

interface ChatInterfaceProps {
  chatId?: string;
  chat?: Chat;
  onMessageAdded?: (newChat: Chat) => void;
}

// Convert ChatMessage to a display format for rendering
interface DisplayMessage {
  role: "user" | "assistant";
  content?: string;
  message?: string | object;
  order?: number;
  created_at?: string;
  courseRecommendations?: any[]; // For course recommendations
}

// Type for the structured course recommendation object
interface CourseRecommendation {
  course_id: string;
  course_code?: string;
  course_title: string;
  course_description?: string;
  career_alignment?: string;
  skill_development?: string[];
  relevance_score?: number;
}

// Parse the LLM response to extract course recommendations
const parseLLMResponse = (responseData: string | object) => {
  try {
    // If already an object, check if it has recommended_courses
    if (typeof responseData === "object" && responseData !== null) {
      const obj = responseData as any;
      if (obj.recommended_courses) {
        return {
          text: obj.response || "",
          recommendations: obj.recommended_courses || [],
          chat_title: obj.chat_title,
        };
      }

      // If it's a response object with json_response
      if (obj.json_response?.recommended_courses) {
        return {
          text: obj.response || "",
          recommendations: obj.json_response.recommended_courses || [],
          chat_title: obj.chat_title,
        };
      }

      return {
        text: JSON.stringify(responseData),
        recommendations: [],
        chat_title: "",
      };
    }

    // Handle string responses
    const responseText = responseData as string;

    // Try to parse as JSON if it's a JSON string
    if (
      responseText.includes('"recommended_courses"') ||
      responseText.includes('"json_response"')
    ) {
      try {
        const jsonData = JSON.parse(responseText);
        return {
          text: jsonData.response || jsonData.text || responseText,
          recommendations:
            jsonData.recommended_courses ||
            jsonData.json_response?.recommended_courses ||
            [],
        };
      } catch (e) {
        // If JSON parsing fails, return the original text
        return {
          text: responseText,
          recommendations: [],
        };
      }
    }

    // If not JSON, return the text as is
    return {
      text: responseText,
      recommendations: [],
    };
  } catch (error) {
    console.error("Error parsing LLM response:", error);

    // Return a safe value if there's an error
    return {
      text:
        typeof responseData === "string"
          ? responseData
          : "Error parsing response",
      recommendations: [],
    };
  }
};

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  chatId,
  chat,
  onMessageAdded,
}) => {
  const [message, setMessage] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [hasUserInteracted, setHasUserInteracted] = useState<boolean>(false);

  // Get the API hooks
  const [getAIResponse] = useGetAIResponseMutation();
  const [addChatMessage] = useAddChatMessageMutation();
  const [updateChat] = useUpdateChatMutation();

  // Get chat messages if chatId is provided
  const {
    data: fetchedMessages,
    isLoading: isFetchingMessages,
    refetch: refetchMessages,
  } = useGetChatMessagesQuery(chatId || "", { skip: !chatId });

  // Get chat details if chatId is provided
  const { data: chatDetails, refetch: refetchChatDetails } =
    useGetChatDetailsQuery(chatId || "", { skip: !chatId });

  // Check if this chat has a pending response
  const { data: isPending, refetch: refetchPendingStatus } =
    useGetChatPendingStatusQuery(chatId || "new-chat", {
      skip: !chatId && !hasUserInteracted,
      pollingInterval: 1000, // Poll every second to check for updates
    });

  // Welcome message
  const welcomeMessage: DisplayMessage = {
    role: "assistant",
    content:
      "Hello! I'm your Course Advisor AI. I can help you find the best courses based on your career goals and interests. Tell me what career path you're interested in, your academic background, or specific skills you want to develop.",
    order: 0,
  };

  // Convert API messages to DisplayMessage format
  const convertMessages = (
    messages: ChatMessage[] | undefined
  ): DisplayMessage[] => {
    if (!messages) return [];

    // Map API messages to display format and sort by creation time
    return messages
      .map((msg) => {
        // Handle different message formats
        try {
          if (msg.role === "assistant") {
            // Check if message has recommended courses
            if (
              (typeof msg.message === "string" &&
                (msg.message.includes("recommended_courses") ||
                  msg.message.includes("json_response"))) ||
              (typeof msg.message === "object" &&
                msg.message !== null &&
                ("recommended_courses" in msg.message ||
                  "json_response" in msg.message))
            ) {
              const parsedResponse = parseLLMResponse(msg.message);
              if (
                parsedResponse.recommendations &&
                parsedResponse.recommendations.length > 0
              ) {
                return {
                  role: msg.role,
                  content: parsedResponse.text,
                  message: msg.message,
                  created_at: msg.created_at,
                  courseRecommendations: parsedResponse.recommendations,
                };
              }
            }
          }

          // For messages without recommendations or user messages
          return {
            role: msg.role,
            content:
              typeof msg.message === "string"
                ? msg.message
                : msg.message !== null && typeof msg.message === "object"
                ? (msg.message as any).response || JSON.stringify(msg.message)
                : String(msg.message),
            message: msg.message,
            created_at: msg.created_at,
          };
        } catch (error) {
          console.error("Error parsing message content:", error);
          return {
            role: msg.role,
            content:
              typeof msg.message === "string"
                ? msg.message
                : "Error displaying message",
            message: msg.message,
            created_at: msg.created_at,
          };
        }
      })
      .sort((a, b) => {
        // Sort by creation time
        return (
          new Date(a.created_at || "").getTime() -
          new Date(b.created_at || "").getTime()
        );
      });
  };

  // Determine if this is a new conversation or existing chat
  const isNewConversation = !chatId && !chat?.id;

  // Get all messages to display, including welcome message for new chats
  const displayMessages =
    chatId && fetchedMessages
      ? convertMessages(fetchedMessages)
      : [welcomeMessage];

  // Get chat title
  const chatTitle = chatDetails?.title || chat?.title || "New Conversation";

  // Add a message count state to force re-renders when messages change
  const [messageCount, setMessageCount] = useState(0);

  // Update message count when messages change
  useEffect(() => {
    setMessageCount(displayMessages.length);
  }, [displayMessages.length]);

  // Scroll to bottom when messages change or when responses are received
  useEffect(() => {
    // First immediate scroll attempt
    scrollToBottom();

    // Then try again after a short delay to ensure all content is rendered
    const scrollTimer = setTimeout(() => {
      scrollToBottom();
    }, 100);

    // And try once more after a longer delay for images/content to load
    const finalScrollTimer = setTimeout(() => {
      scrollToBottom();
    }, 500);

    return () => {
      clearTimeout(scrollTimer);
      clearTimeout(finalScrollTimer);
    };
  }, [messageCount, isPending, fetchedMessages]);

  // Improved scrollToBottom function
  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  const [addCourseToMyCourses, { isLoading: isAddingCourse }] =
    useAddCourseToMyCoursesMutation();

  // Handler for "Add to My Courses" button
  const handleAddCourse = async (course: CourseRecommendation) => {
    try {
      // Add course to My Courses
      console.log(course);
      const result = await addCourseToMyCourses({
        courseId: course.course_id,
        course: course,
      }).unwrap();

      // Show success notification
      alert(`Course "${course.course_title}" has been added to your courses!`);
    } catch (error) {
      // Handle any errors during course addition
      console.error("Failed to add course:", error);
      alert(
        `Failed to add course. ${
          error instanceof Error ? error.message : "Please try again."
        }`
      );
    }
  };

  // Get a suggested title based on user's first message
  const getSuggestedTitle = (userMessage: string): string => {
    // Take the first 50 characters of the message
    let title = userMessage.substring(0, 50).trim();

    // If the title is too short, add a generic suffix
    if (title.length < 10) {
      title += " conversation";
    }

    // Add ellipsis if truncated
    if (userMessage.length > 50) {
      title += "...";
    }

    return title;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() === "" || isLoading || isPending) return;

    const userMessage = message.trim();
    const isFirstMessage = displayMessages.length === 1 || isNewConversation;

    // Update UI immediately
    setMessage("");
    setIsLoading(true);
    setHasUserInteracted(true);

    try {
      // First send user message
      const userApiResponse = await addChatMessage({
        // Only include chatId if this is an existing conversation
        ...(!isNewConversation && chatId && { chatId }),
        message: userMessage,
        role: "user",
        title: isFirstMessage ? getSuggestedTitle(userMessage) : undefined,
      }).unwrap();

      // If this is the first message, update the chat title
      //   if (isFirstMessage) {

      //   }

      // Get user ID from localStorage
      const user = localStorage.getItem("user");
      const parsedUser = user ? JSON.parse(user) : null;
      const userId = parsedUser?.id || "anonymous";

      // Set pending status manually (API will also set it)
      pendingResponses.setPending(userApiResponse.chat.id, true);

      // Refetch messages to get the user message and to check pending status
      refetchMessages();
      refetchPendingStatus();

      // Call external AI API to generate response
      const aiResponse = await getAIResponse({
        user_id: userId,
        query: userMessage,
        session_id: userApiResponse.chat.id || null,
      }).unwrap();

      // Then send assistant response
      const assistantMessageResponse = await addChatMessage({
        chatId: userApiResponse.chat.id, // Use the chat ID from the user message response
        message: aiResponse, // The external API response
        role: "assistant",
      }).unwrap();

      pendingResponses.setPending(userApiResponse.chat.id, false);

      // Update chat title based on first AI response for new chats
      //   if (isFirstMessage) {
      // Try to extract a meaningful title from the AI response
      const parsedResponse = parseLLMResponse(aiResponse);
      const recommendationTitle = parsedResponse.chat_title || null;

      // Check if the AI response contains a chat_title
      const chatTitleFromResponse =
        typeof aiResponse === "object" && (aiResponse as any).chat_title;

      const newTitle =
        chatTitleFromResponse ||
        recommendationTitle ||
        getSuggestedTitle(parsedResponse.text) ||
        chatDetails?.title ||
        getSuggestedTitle(userMessage);

      try {
        await updateChat({
          id: userApiResponse.chat.id,
          title: newTitle,
        });
      } catch (error) {
        console.error("Failed to update chat title from AI response:", error);
      }
      //   }

      // Refetch messages and chat details to get the AI response
      refetchMessages();
      refetchChatDetails();
      refetchPendingStatus();

      // Call onMessageAdded with the chat ID
      if (onMessageAdded) {
        onMessageAdded({
          id: userApiResponse.chat.id,
          title: chatDetails?.title || getSuggestedTitle(userMessage),
          userId: userApiResponse.chat.userId,
          created_at: userApiResponse.chat.created_at,
        });
      }
    } catch (error) {
      console.error("Failed to send message", error);

      // Clear pending status on error
      if (chatId) {
        pendingResponses.setPending(chatId, false);
        refetchPendingStatus();
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Render skill tags for each course
  const renderSkills = (skills: string[] = []) => {
    return skills.map((skill, idx) => (
      <span
        key={idx}
        className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2 mt-1 inline-block"
      >
        {skill}
      </span>
    ));
  };

  // Render course recommendation card
  const renderCourseCard = (course: any) => {
    return (
      <div
        key={course.course_id}
        className="border-l-4 border-indigo-500 pl-4 mb-4 pb-4"
      >
        <div className="flex justify-between items-start">
          <h4 className="font-medium text-indigo-800">
            {course.course_code || course.course_id}: {course.course_title}
          </h4>
          <button
            onClick={() => handleAddCourse(course)}
            className="bg-indigo-600 hover:bg-indigo-700 text-white text-xs px-3 py-1 rounded transition"
          >
            Add to My Courses
          </button>
        </div>
        <p className="text-sm text-gray-700 my-2">
          {course.course_description || "No description available"}
        </p>
        <div className="mt-2">
          {course.career_alignment && (
            <>
              <div className="text-xs text-gray-500 mb-1">
                Career alignment:
              </div>
              <div className="text-sm text-indigo-700 font-medium mb-2">
                {course.career_alignment}
              </div>
            </>
          )}
          {course.skill_development && course.skill_development.length > 0 && (
            <>
              <div className="text-xs text-gray-500 mb-1">
                Skills you'll develop:
              </div>
              <div className="flex flex-wrap">
                {renderSkills(course.skill_development)}
              </div>
            </>
          )}
        </div>
      </div>
    );
  };

  // Render markdown text with proper formatting
  const renderMarkdown = (text: string) => {
    // Simple markdown parsing for headers, bold, etc.
    const formattedText = text
      .replace(/# (.*?)(\n|$)/g, '<h1 class="text-2xl font-bold my-3">$1</h1>')
      .replace(/## (.*?)(\n|$)/g, '<h2 class="text-xl font-bold my-2">$1</h2>')
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n\n/g, '</p><p class="mb-2">')
      .replace(/\n(\d+\. )/g, '</p><p class="mb-1">$1');

    return (
      <div
        dangerouslySetInnerHTML={{
          __html: `<p class="mb-2">${formattedText}</p>`,
        }}
      />
    );
  };

  // Get message content safely, handling both string and object formats
  const getMessageContent = (msg: DisplayMessage): string => {
    if (msg.content) return msg.content;

    if (msg.message) {
      if (typeof msg.message === "string") {
        return msg.message;
      } else if (typeof msg.message === "object" && msg.message !== null) {
        return (msg.message as any).response || JSON.stringify(msg.message);
      }
    }

    return "No content available";
  };

  // Render typing indicator
  const renderTypingIndicator = () => {
    return (
      <div className="typing-indicator-container">
        <div className="typing-indicator">
          <span className="dot"></span>
          <span className="dot"></span>
          <span className="dot"></span>
        </div>
      </div>
    );
  };

  return (
    <div className="main-content">
      <div className="top-nav">
        <div className="breadcrumb">
          <span>Conversations</span>
          <span className="separator">/</span>
          <span className="current-page">{chatTitle}</span>
        </div>
      </div>

      <div className="messages-area">
        {isFetchingMessages ? (
          <div className="loading-messages">
            <div className="typing-indicator">
              <div className="dot dot1"></div>
              <div className="dot dot2"></div>
              <div className="dot dot3"></div>
            </div>
          </div>
        ) : (
          <>
            {displayMessages.map((msg, index) => {
              const messageContent = getMessageContent(msg);
              const isAssistant = msg.role === "assistant";
              const formattedTime = msg.created_at
                ? new Date(msg.created_at).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                : "";

              return (
                <div
                  key={index}
                  className={`message-row ${
                    isAssistant ? "message-assistant" : "message-user"
                  }`}
                >
                  {/* Message Content */}
                  <div className="message-container">
                    <div className="message-header">
                      {!isAssistant && (
                        <>
                          <div className="avatar-container">
                            <div className="message-avatar user">U</div>
                          </div>
                          <div className="message-sender">You</div>
                          <div className="message-time">{formattedTime}</div>
                        </>
                      )}

                      {isAssistant && (
                        <>
                          <div className="message-time">{formattedTime}</div>
                          <div className="message-sender">Course Advisor</div>
                          <div className="avatar-container">
                            <div className="message-avatar assistant">CA</div>
                          </div>
                        </>
                      )}
                    </div>
                    <div className="message-content">
                      <div className="message-text">
                        {messageContent && messageContent.includes("#")
                          ? renderMarkdown(messageContent)
                          : messageContent}
                      </div>

                      {/* Course Recommendations */}
                      {msg.courseRecommendations &&
                        msg.courseRecommendations.length > 0 && (
                          <div className="course-recommendations mt-4 bg-white rounded-lg p-4 shadow-sm">
                            <h3 className="text-lg font-medium mb-3">
                              Recommended Courses
                            </h3>
                            <div className="course-list">
                              {msg.courseRecommendations.map((course) =>
                                renderCourseCard(course)
                              )}
                            </div>
                          </div>
                        )}
                    </div>
                  </div>
                </div>
              );
            })}

            {/* Typing indicator when AI is responding */}
            {isPending && (
              <div className="message-row message-assistant">
                <div className="message-container">
                  <div className="message-header">
                    <div className="message-time"></div>
                    <div className="message-sender">Course Advisor</div>
                    <div className="avatar-container">
                      <div className="message-avatar assistant">CA</div>
                    </div>
                  </div>
                  <div className="message-content">
                    {renderTypingIndicator()}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef}></div>
      </div>

      <div className="input-area">
        <div className="input-container">
          <form onSubmit={handleSubmit}>
            <div className="input-wrapper">
              <input
                type="text"
                placeholder="Ask about courses, career paths, and academic advice..."
                className="message-input"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                disabled={isLoading || isPending}
              />
              <div className="input-actions">
                <button
                  type="submit"
                  className="send-button"
                  disabled={isLoading || isPending || message.trim() === ""}
                >
                  <span className="send-icon">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086l-1.414 4.926a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z" />
                    </svg>
                  </span>
                </button>
              </div>
            </div>
          </form>
          <div className="input-footer">
            <span className="hint-text">
              Ask specific questions for better results
            </span>
            <span className="model-info">Course Advisor AI v1.0</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
