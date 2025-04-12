import React, { useState, useEffect } from "react";
import { User } from "../types";
import Logo from "./Logo";
import {
  useGetUserChatsQuery,
  useCreateChatMutation,
} from "../services/chat_api";
import { useAppDispatch } from "../hooks/redux";
import { clearCredentials } from "../store/slices/authSlice";

interface SidebarProps {
  user: User;
  onViewChange: (view: string) => void;
  onSelectChat: (chatId: string) => void;
  currentChatId: string | null;
}

const Sidebar: React.FC<SidebarProps> = ({
  user,
  onViewChange,
  onSelectChat,
  currentChatId,
}) => {
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const [isCreatingChat, setIsCreatingChat] = useState(false);

  // Redux dispatcher for logout
  const dispatch = useAppDispatch();

  // Fetch chats using RTK Query
  const {
    data: chats = [],
    isLoading: isChatsLoading,
    refetch: refetchChats,
  } = useGetUserChatsQuery();

  // Create chat mutation
  const [createChat] = useCreateChatMutation();

  // Handle creating a new chat
  const handleCreateNewChat = async () => {
    if (isCreatingChat) return;

    setIsCreatingChat(true);
    try {
      // Create a new chat via API
      const result = await createChat({
        title: "New Conversation",
      }).unwrap();

      // Refetch the chat list
      await refetchChats();

      // Select the new chat and change to chat view
      onViewChange("chat");
      onSelectChat(result.id);
    } catch (error) {
      console.error("Failed to create new chat:", error);
    } finally {
      setIsCreatingChat(false);
    }
  };

  // Handle course catalog click
  const handleCourseCatalogClick = () => {
    onViewChange("courses");
  };

  // Handle my courses click
  const handleMyCoursesClick = () => {
    onViewChange("my_courses");
  };

  // Handle logout
  const handleLogout = () => {
    dispatch(clearCredentials());
    setIsUserMenuOpen(false);
  };

  // Toggle user menu
  const toggleUserMenu = () => {
    setIsUserMenuOpen(!isUserMenuOpen);
  };

  // Format chat name based on messages
  const getChatName = (chat: any): string => {
    if (chat.title && chat.title !== "New Conversation") {
      return chat.title;
    }

    // If there are messages, use the first question as the title
    if (chat.messages && chat.messages.length > 0) {
      const firstQuestion = chat.messages[0].question;
      return firstQuestion.length > 25
        ? firstQuestion.substring(0, 25) + "..."
        : firstQuestion;
    }

    // Fallback title
    return "New Conversation";
  };

  // Sort chats by latest first
  const sortedChats = [...chats].sort(
    (a, b) =>
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );

  return (
    <div className="sidebar">
      <div className="logo-container">
        <Logo textColor="white" />
      </div>

      <button
        className={`new-chat-button ${isCreatingChat ? "opacity-70" : ""}`}
        onClick={handleCreateNewChat}
        disabled={isCreatingChat}
      >
        <span className="plus-icon">{isCreatingChat ? "..." : "+"}</span>
        New chat
      </button>

      {/* Course Catalog */}
      <div className="sidebar-link" onClick={handleCourseCatalogClick}>
        <svg
          className="sidebar-icon"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <rect
            x="4"
            y="4"
            width="16"
            height="16"
            rx="2"
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M4 8H20"
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
          />
        </svg>
        <span>Course Catalog</span>
      </div>

      {/* My Courses */}
      <div className="sidebar-link" onClick={handleMyCoursesClick}>
        <svg
          className="sidebar-icon"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <rect
            x="3"
            y="3"
            width="18"
            height="18"
            rx="3"
            stroke="white"
            strokeWidth="2"
          />
          <circle cx="8" cy="10" r="1" fill="white" />
          <circle cx="12" cy="10" r="1" fill="white" />
          <circle cx="16" cy="10" r="1" fill="white" />
          <path
            d="M7 14H17"
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
          />
        </svg>
        <span>My Courses</span>
      </div>

      {/* Recents Section */}
      <div className="recents-section">
        <div className="recents-header">RECENTS</div>
        <div className="chat-list">
          {isChatsLoading ? (
            <div className="chat-loading">Loading chats...</div>
          ) : sortedChats.length > 0 ? (
            sortedChats.map((chat) => (
              <div
                key={chat.id}
                className={`chat-item ${
                  chat.id === currentChatId ? "active" : ""
                }`}
                onClick={() => {
                  onViewChange("chat");
                  onSelectChat(chat.id);
                }}
              >
                <span className="chat-name">{getChatName(chat)}</span>
                <span className="more-icon">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    width="16"
                    height="16"
                  >
                    <path d="M6 10a2 2 0 11-4 0 2 2 0 014 0zM12 10a2 2 0 11-4 0 2 2 0 014 0zM16 12a2 2 0 100-4 2 2 0 000 4z" />
                  </svg>
                </span>
              </div>
            ))
          ) : (
            <div className="no-chats">
              No recent conversations. Start a new one!
            </div>
          )}
        </div>
      </div>

      {/* User profile with dropdown */}
      <div className="user-profile">
        <div
          className="profile-info"
          onClick={toggleUserMenu}
          role="button"
          aria-haspopup="menu"
          aria-expanded={isUserMenuOpen}
        >
          <div className="avatar">
            {user.firstName ? user.firstName.charAt(0).toUpperCase() : "D"}
          </div>
          <div className="user-name">{user.firstName || "Demo"}</div>
          <svg
            className={`dropdown-arrow ${isUserMenuOpen ? "open" : ""}`}
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M6 9L12 15L18 9"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        {isUserMenuOpen && (
          <div className="user-menu">
            <button onClick={handleLogout} className="menu-item">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M16 17L21 12L16 7"
                  stroke="white"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M21 12H9"
                  stroke="white"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M9 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H9"
                  stroke="white"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <span>Logout</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
