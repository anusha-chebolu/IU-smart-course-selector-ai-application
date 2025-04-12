import React, { useState, useEffect } from "react";
import { useAppSelector } from "../hooks/redux";
import Sidebar from "./Sidebar_new";
import ChatInterface from "./chat/ChatInterface";
import CourseSearch from "./courses/CourseSearch";
import { User } from "../types";
import MyCourses from "./courses/MyCourses";

const Dashboard: React.FC = () => {
  const { user } = useAppSelector((state) => state.auth);

  // State to track which view is active - default to courses view
  const [activeView, setActiveView] = useState<string>("courses");

  // Current active chat ID - this will be controlled by Sidebar
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);

  const defaultUser: User = {
    id: "",
    username: "",
    firstName: "",
    lastName: "",
    email: "",
  };

  // Set the current user from Redux auth state
  const [currentUser, setCurrentUser] = useState<User>(user || defaultUser);

  // Update currentUser when the Redux state changes
  useEffect(() => {
    if (user) {
      setCurrentUser(user);
    }
  }, [user]);

  // Handle view switching
  const handleViewChange = (view: string) => {
    setActiveView(view);
  };

  // Handle chat selection from Sidebar
  const handleChatSelection = (chatId: string) => {
    setCurrentChatId(chatId);
    setActiveView("chat");
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar handles its own API calls and state management */}
      <Sidebar
        user={currentUser}
        onViewChange={handleViewChange}
        onSelectChat={handleChatSelection}
        currentChatId={currentChatId || ""}
      />

      {/* Main content area */}
      <div className="main-content">
        {activeView === "chat" && currentChatId ? (
          <ChatInterface chatId={currentChatId || ""} />
        ) : activeView === "my_courses" ? (
          <MyCourses />
        ) : (
          <CourseSearch />
        )}
      </div>
    </div>
  );
};

export default Dashboard;
