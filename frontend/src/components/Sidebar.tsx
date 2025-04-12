import React, { useState } from "react";
import { Chat, User } from "../types";
import Logo from "./Logo";

interface SidebarProps {
  chats: Chat[];
  currentChatId: string;
  onSelectChat: (chatId: string) => void;
  onCreateNewChat: () => void;
  onLogout: () => void;
  user: User;
  onCourseCatalogClick: () => void;
  onMyCoursesClick: () => void;
  activeView?: string;
}

const Sidebar: React.FC<SidebarProps> = ({
  chats,
  currentChatId,
  onSelectChat,
  onCreateNewChat,
  onLogout,
  user,
  onCourseCatalogClick,
  onMyCoursesClick,
  activeView = "",
}) => {
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);

  const toggleUserMenu = () => {
    setIsUserMenuOpen(!isUserMenuOpen);
  };

  return (
    <div className="sidebar">
      <div className="logo-container">
        <Logo textColor="white" />
      </div>

      <button className="new-chat-button" onClick={onCreateNewChat}>
        <span className="plus-icon">+</span>
        New chat
      </button>

      {/* Course Catalog */}
      <div
        className={`sidebar-link ${
          activeView === "course_catalog" ? "active" : ""
        }`}
        onClick={onCourseCatalogClick}
      >
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
      <div
        className={`sidebar-link ${
          activeView === "my_courses" ? "active" : ""
        }`}
        onClick={onMyCoursesClick}
      >
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
          {chats.map((chat) => (
            <div
              key={chat.id}
              className={`chat-item ${
                chat.id === currentChatId ? "active" : ""
              }`}
              onClick={() => onSelectChat(String(chat.id))}
            >
              <span className="chat-name">{chat.name}</span>
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
          ))}
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
            <button onClick={onLogout} className="menu-item">
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
