// // // components/AppContent.tsx
// // import React, { useState, useEffect } from "react";
// // import { Routes, Route, Navigate } from "react-router-dom";
// // import { useAppSelector, useAppDispatch } from "../hooks/redux";
// // import { setCredentials, clearCredentials } from "../store/slices/authSlice";
// // import Sidebar from "./Sidebar";
// // import ChatInterface from "./ChatInterface";
// // import LoginPageWithSlides from "./LoginPage";
// // import ProtectedRoute from "./ProtectedRoute";
// // import CourseSearch from "./courses/CourseSearch";
// // import { Chat, Message, User } from "../types";

// // const AppContent: React.FC = () => {
// //   const { isAuthenticated, user } = useAppSelector((state) => state.auth);
// //   const dispatch = useAppDispatch();

// //   // State to track which view is active
// //   const [activeView, setActiveView] = useState<"chat" | "courses">("chat");

// //   const defaultUser: User = {
// //     id: "",
// //     username: "",
// //     firstName: "",
// //     lastName: "",
// //     email: "",
// //   };

// //   // Set the current user from Redux auth state
// //   const [currentUser, setCurrentUser] = useState<User>(user || defaultUser);

// //   // Update currentUser when the Redux state changes
// //   useEffect(() => {
// //     if (user) {
// //       setCurrentUser(user);
// //     }
// //   }, [user]);

// //   // Initial welcome message
// //   const welcomeMessage: Message = {
// //     role: "assistant",
// //     content:
// //       "Hello! I'm your Course Advisor AI. I can help you find the best courses based on your career goals and interests. Tell me what career path you're interested in, your academic background, or specific skills you want to develop.",
// //   };

// //   // Sample chat history
// //   const [chats, setChats] = useState<Chat[]>([
// //     {
// //       id: 1,
// //       name: "Software Engineering Path",
// //       messages: [welcomeMessage],
// //     },
// //     {
// //       id: 2,
// //       name: "Data Science Career Advice",
// //       messages: [welcomeMessage],
// //     },
// //     {
// //       id: 3,
// //       name: "UX Design Courses",
// //       messages: [welcomeMessage],
// //     },
// //     {
// //       id: 4,
// //       name: "Business Analytics Track",
// //       messages: [welcomeMessage],
// //     },
// //     {
// //       id: 5,
// //       name: "Cybersecurity Specialization",
// //       messages: [welcomeMessage],
// //     },
// //     {
// //       id: 6,
// //       name: "Machine Learning Focus",
// //       messages: [welcomeMessage],
// //     },
// //     {
// //       id: 7,
// //       name: "Web Development Roadmap",
// //       messages: [welcomeMessage],
// //     },
// //   ]);

// //   // Current active chat
// //   const [currentChatId, setCurrentChatId] = useState<number>(1);

// //   // Find current chat based on ID
// //   const currentChat =
// //     chats.find((chat) => chat.id === currentChatId) || chats[0];

// //   // Function to create a new chat
// //   const createNewChat = () => {
// //     // Create a new ID by finding the maximum current ID and adding 1
// //     const newChatId = Math.max(...chats.map((chat) => chat.id)) + 1;

// //     const newChat: Chat = {
// //       id: newChatId,
// //       name: "New Conversation",
// //       messages: [welcomeMessage],
// //     };

// //     setChats([newChat, ...chats]);
// //     setCurrentChatId(newChatId);
// //     // When creating a new chat, switch to chat view
// //     setActiveView("chat");
// //   };

// //   // Handle chat selection
// //   const selectChat = (chatId: number) => {
// //     setCurrentChatId(chatId);
// //     // When selecting a chat, switch to chat view
// //     setActiveView("chat");
// //   };

// //   // Handle navigation to Course Catalog
// //   const handleCourseCatalogClick = () => {
// //     setActiveView("courses");
// //   };

// //   // Handle navigation to My Courses
// //   const handleMyCoursesClick = () => {
// //     // For now, this can also navigate to the course search
// //     setActiveView("courses");
// //   };

// //   // Handle new message and AI response
// //   const handleNewMessage = (message: string, updateChatName = false) => {
// //     const userMessage: Message = { role: "user", content: message };

// //     // Update the messages in the current chat
// //     let updatedChats = chats.map((chat) => {
// //       if (chat.id === currentChatId) {
// //         // If this is the first user message and updateChatName is true, update the chat name
// //         const updatedName =
// //           updateChatName && chat.name === "New Conversation"
// //             ? message.length > 25
// //               ? message.substring(0, 25) + "..."
// //               : message
// //             : chat.name;

// //         return {
// //           ...chat,
// //           name: updatedName,
// //           messages: [...chat.messages, userMessage],
// //         };
// //       }
// //       return chat;
// //     });

// //     setChats(updatedChats);

// //     // Return a simulated AI response after a delay
// //     return new Promise<void>((resolve) => {
// //       setTimeout(() => {
// //         const aiResponse = generateResponse(message);
// //         const responseMessage: Message = {
// //           role: "assistant",
// //           content: aiResponse,
// //         };

// //         const responsedChats = updatedChats.map((chat) => {
// //           if (chat.id === currentChatId) {
// //             return {
// //               ...chat,
// //               messages: [...chat.messages, userMessage, responseMessage],
// //             };
// //           }
// //           return chat;
// //         });

// //         setChats(responsedChats);
// //         resolve();
// //       }, 1500);
// //     });
// //   };

// //   // Function to handle login
// //   const handleLogin = async (
// //     username: string,
// //     password: string
// //   ): Promise<boolean> => {
// //     // In a real app, you would validate credentials with a backend
// //     return new Promise((resolve) => {
// //       // Simulate network delay
// //       setTimeout(() => {
// //         if (username && password) {
// //           // Simulate successful login with mock user data
// //           const userData = {
// //             id: "user123",
// //             username: username,
// //             firstName: "Demo",
// //             lastName: "User",
// //             email: "demo@example.com",
// //           };

// //           // Update Redux state
// //           dispatch(
// //             setCredentials({
// //               user: userData,
// //               token: `mock-token-${Date.now()}`,
// //             })
// //           );

// //           resolve(true);
// //         } else {
// //           resolve(false);
// //         }
// //       }, 500);
// //     });
// //   };

// //   // Function to handle signup with extended user info
// //   const handleSignup = async (
// //     username: string,
// //     password: string,
// //     firstName: string,
// //     lastName: string,
// //     email: string
// //   ): Promise<boolean> => {
// //     // In a real app, you would send this data to your backend
// //     return new Promise((resolve) => {
// //       // Simulate network delay
// //       setTimeout(() => {
// //         if (username && password && firstName && lastName && email) {
// //           // Generate a mock user ID (in a real app, this would come from the backend)
// //           const id = `user_${Math.floor(Math.random() * 10000)}`;

// //           // Create user data
// //           const userData = {
// //             id,
// //             username,
// //             firstName,
// //             lastName,
// //             email,
// //           };

// //           // Update Redux state
// //           dispatch(
// //             setCredentials({
// //               user: userData,
// //               token: `mock-token-${Date.now()}`,
// //             })
// //           );

// //           resolve(true);
// //         } else {
// //           resolve(false);
// //         }
// //       }, 800);
// //     });
// //   };

// //   // Handle logout
// //   const handleLogout = () => {
// //     dispatch(clearCredentials());
// //   };

// //   // Function to generate relevant course recommendations based on input
// //   const generateResponse = (userInput: string): string => {
// //     const input = userInput.toLowerCase();

// //     if (
// //       input.includes("software") ||
// //       input.includes("programming") ||
// //       input.includes("developer")
// //     ) {
// //       return `Based on your interest in software development, here are some recommended courses:

// // 1. **CS101: Introduction to Computer Science** - Foundational concepts in computing
// // 2. **CS201: Data Structures and Algorithms** - Essential for coding interviews
// // 3. **CS350: Software Engineering Principles** - Learn industry best practices
// // 4. **WEB101: Full-Stack Web Development** - Practical skills for web applications

// // These courses will help build a strong foundation for a software engineering career. Would you like more specialized recommendations in any particular area of software development?`;
// //     } else if (
// //       input.includes("data") ||
// //       input.includes("analytics") ||
// //       input.includes("machine learning")
// //     ) {
// //       return `For a career in data science or analytics, I recommend these courses:

// // 1. **STAT101: Statistics for Data Analysis** - Build your foundation
// // 2. **DS201: Introduction to Data Science** - Learn core concepts and tools
// // 3. **ML301: Machine Learning Fundamentals** - Understand algorithms and models
// // 4. **DS450: Big Data Technologies** - Work with large-scale data systems

// // Would you like me to suggest more specialized courses based on particular industries you're interested in?`;
// //     } else if (
// //       input.includes("design") ||
// //       input.includes("ux") ||
// //       input.includes("ui")
// //     ) {
// //       return `For user experience and design, these courses would be valuable:

// // 1. **DES101: Design Principles** - Fundamentals of good design
// // 2. **UX201: User Experience Fundamentals** - Learn user-centered design methods
// // 3. **UI301: User Interface Design** - Create effective and attractive interfaces
// // 4. **DES401: Design Systems** - Build scalable design frameworks

// // These will help you build the skills needed for a UX/UI design career. Would you like recommendations for portfolio-building courses as well?`;
// //     } else {
// //       return `Thank you for sharing your interests in "${userInput}". To provide the most relevant course recommendations, I'd like to know a bit more:

// // 1. What specific career outcome are you hoping for?
// // 2. Do you already have experience in this field?
// // 3. Are you looking for introductory courses or more advanced specializations?

// // This will help me suggest the best academic path for your goals.`;
// //     }
// //   };

// //   // Define the main dashboard content
// //   const DashboardContent = () => (
// //     <div className="course-advisor-container">
// //       <Sidebar
// //         chats={chats}
// //         currentChatId={currentChatId}
// //         onSelectChat={selectChat}
// //         onCreateNewChat={createNewChat}
// //         user={currentUser}
// //         onLogout={handleLogout}
// //         onCourseCatalogClick={handleCourseCatalogClick}
// //         onMyCoursesClick={handleMyCoursesClick}
// //       />

// //       {/* Conditionally render either ChatInterface or CourseSearch based on activeView */}
// //       {activeView === "chat" ? (
// //         <ChatInterface chat={currentChat} onSendMessage={handleNewMessage} />
// //       ) : (
// //         <CourseSearch />
// //       )}
// //     </div>
// //   );

// //   return (
// //     <Routes>
// //       <Route
// //         path="/login"
// //         element={
// //           isAuthenticated ? (
// //             <Navigate to="/dashboard" replace />
// //           ) : (
// //             <LoginPageWithSlides onSignup={handleSignup} />
// //           )
// //         }
// //       />
// //       <Route
// //         path="/dashboard"
// //         element={
// //           <ProtectedRoute>
// //             <DashboardContent />
// //           </ProtectedRoute>
// //         }
// //       />
// //       <Route
// //         path="/"
// //         element={
// //           <Navigate to={isAuthenticated ? "/dashboard" : "/login"} replace />
// //         }
// //       />
// //       <Route
// //         path="*"
// //         element={
// //           <Navigate to={isAuthenticated ? "/dashboard" : "/login"} replace />
// //         }
// //       />
// //     </Routes>
// //   );
// // };

// // export default AppContent;

// import React, { useState, useEffect } from "react";
// import { Routes, Route, Navigate } from "react-router-dom";
// import { useAppSelector, useAppDispatch } from "../hooks/redux";
// import { setCredentials, clearCredentials } from "../store/slices/authSlice";
// import Sidebar from "./Sidebar";
// import ChatInterface from "./ChatInterface";
// import LoginPageWithSlides from "./LoginPage";
// import ProtectedRoute from "./ProtectedRoute";
// import CourseSearch from "./courses/CourseSearch";
// import { Chat, Message, User } from "../types";
// import { useGetUserChatsQuery } from "../services/chat_api";
// import { generateResponse } from "./generateResponse";
// const AppContent: React.FC = () => {
//   const { isAuthenticated, user } = useAppSelector((state) => state.auth);
//   const dispatch = useAppDispatch();

//   // State to track which view is active
//   const [activeView, setActiveView] = useState<"chat" | "courses">("chat");

//   // Fetch user's chats
//   const { data: userChats = [], isLoading: isChatsLoading } =
//     useGetUserChatsQuery(undefined, {
//       skip: !isAuthenticated, // Only fetch if authenticated
//     });

//   const defaultUser: User = {
//     id: "",
//     username: "",
//     firstName: "",
//     lastName: "",
//     email: "",
//   };

//   // Set the current user from Redux auth state
//   const [currentUser, setCurrentUser] = useState<User>(user || defaultUser);

//   // Update currentUser when the Redux state changes
//   useEffect(() => {
//     if (user) {
//       setCurrentUser(user);
//     }
//   }, [user]);

//   // Initial welcome message
//   const welcomeMessage: Message = {
//     role: "assistant",
//     content:
//       "Hello! I'm your Course Advisor AI. I can help you find the best courses based on your career goals and interests. Tell me what career path you're interested in, your academic background, or specific skills you want to develop.",
//   };

//   // Transform UserChat to Chat
//   const transformedChats: Chat[] = userChats.map((userChat, index) => ({
//     id: index + 1, // Generate a numeric ID
//     name:
//       userChat.messages.length > 0
//         ? userChat.messages[0].question.length > 25
//           ? userChat.messages[0].question.substring(0, 25) + "..."
//           : userChat.messages[0].question
//         : "New Conversation",
//     messages: userChat.messages
//       .map((msg) => ({
//         role: "user",
//         content: msg.question,
//       }))
//       .concat({
//         role: "assistant",
//         content:
//           userChat.messages.length > 0
//             ? userChat.messages[0].response
//             : welcomeMessage.content,
//       }),
//   }));

//   // Current active chat
//   const [currentChatId, setCurrentChatId] = useState<number>(
//     transformedChats.length > 0 ? transformedChats[0].id : 1
//   );

//   // Find current chat based on ID
//   const currentChat =
//     transformedChats.find((chat) => chat.id === currentChatId) ||
//     transformedChats[0];

//   // Function to create a new chat
//   const createNewChat = () => {
//     // Create a new ID by finding the maximum current ID and adding 1
//     const newChatId =
//       transformedChats.length > 0
//         ? Math.max(...transformedChats.map((chat) => chat.id)) + 1
//         : 1;

//     const newChat: Chat = {
//       id: newChatId,
//       name: "New Conversation",
//       messages: [welcomeMessage],
//     };

//     // When creating a new chat, switch to chat view
//     setCurrentChatId(newChatId);
//     setActiveView("chat");
//   };

//   // Handle chat selection
//   const selectChat = (chatId: number) => {
//     setCurrentChatId(chatId);
//     // When selecting a chat, switch to chat view
//     setActiveView("chat");
//   };

//   // Handle navigation to Course Catalog
//   const handleCourseCatalogClick = () => {
//     setActiveView("courses");
//   };

//   // Handle navigation to My Courses
//   const handleMyCoursesClick = () => {
//     // For now, this can also navigate to the course search
//     setActiveView("courses");
//   };

//   // Handle new message and AI response
//   const handleNewMessage = (message: string, updateChatName = false) => {
//     const userMessage: Message = { role: "user", content: message };

//     // Return a simulated AI response after a delay
//     return new Promise<void>((resolve) => {
//       setTimeout(() => {
//         const aiResponse = generateResponse(message);
//         const responseMessage: Message = {
//           role: "assistant",
//           content: aiResponse,
//         };

//         resolve();
//       }, 1500);
//     });
//   };

//   // Function to handle login
//   const handleLogin = async (
//     username: string,
//     password: string
//   ): Promise<boolean> => {
//     // In a real app, you would validate credentials with a backend
//     return new Promise((resolve) => {
//       // Simulate network delay
//       setTimeout(() => {
//         if (username && password) {
//           // Simulate successful login with mock user data
//           const userData = {
//             id: "user123",
//             username: username,
//             firstName: "Demo",
//             lastName: "User",
//             email: "demo@example.com",
//           };

//           // Update Redux state
//           dispatch(
//             setCredentials({
//               user: userData,
//               token: `mock-token-${Date.now()}`,
//             })
//           );

//           resolve(true);
//         } else {
//           resolve(false);
//         }
//       }, 500);
//     });
//   };

//   // Function to handle signup with extended user info
//   const handleSignup = async (
//     username: string,
//     password: string,
//     firstName: string,
//     lastName: string,
//     email: string
//   ): Promise<boolean> => {
//     // In a real app, you would send this data to your backend
//     return new Promise((resolve) => {
//       // Simulate network delay
//       setTimeout(() => {
//         if (username && password && firstName && lastName && email) {
//           // Generate a mock user ID (in a real app, this would come from the backend)
//           const id = `user_${Math.floor(Math.random() * 10000)}`;

//           // Create user data
//           const userData = {
//             id,
//             username,
//             firstName,
//             lastName,
//             email,
//           };

//           // Update Redux state
//           dispatch(
//             setCredentials({
//               user: userData,
//               token: `mock-token-${Date.now()}`,
//             })
//           );

//           resolve(true);
//         } else {
//           resolve(false);
//         }
//       }, 800);
//     });
//   };

//   // Handle logout
//   const handleLogout = () => {
//     dispatch(clearCredentials());
//   };

//   // Function to generate relevant course recommendations based on input
//   const generateResponse = (userInput: string): string => {
//     const input = userInput.toLowerCase();

//     // Specific use case for machine learning (using the detailed format)
//     if (input.includes("machine learning") || input.includes("ml")) {
//       return `# Machine Learning Course Recommendations

// I'd be happy to recommend some courses to help you pursue a career in Machine Learning! Based on your interest, here are five excellent courses that would create a strong foundation:

// ## Core Recommendations

// 1. **CSCI-P 556: Applied Machine Learning**
//    This hands-on course focuses on practical implementation rather than just theory, teaching you to apply ML algorithms to real-world data sets. It's perfect for building the fundamental skills employers look for in ML engineers.

// 2. **ENGR-E 533: Deep Learning Systems**
//    This advanced course covers the complete pipeline for building state-of-the-art deep learning systems, including GPU acceleration, parallelization techniques, and deploying models to low-powered hardware.

// 3. **CSCI-B 551: Elements of Artificial Intelligence**
//    This provides the theoretical foundation that underpins machine learning, covering problem-solving, knowledge representation, reasoning under uncertainty, and planning algorithms.

// 4. **ENGR-E 513: Engineering Compilers**
//    While not directly about ML, this systems-level course will help you understand code optimization, which is increasingly valuable for implementing efficient ML systems at scale.

// 5. **BUS-F 534: Fintech Apps in Machine Learning**
//    This specialized course applies ML techniques like LASSO, logistic regressions, and random forests to financial problems, opening doors to one of the highest-paying sectors for ML engineers.

// ## Suggested Learning Path

// I'd recommend starting with Applied Machine Learning to build practical skills, then taking Elements of AI to strengthen your theoretical understanding. From there, progress to Deep Learning Systems for more advanced implementation techniques. The compiler course and fintech application course can be taken later to specialize your knowledge.

// Don't forget to supplement these courses with personal projects that demonstrate your ability to build end-to-end ML systems, as a strong portfolio will be crucial when applying for jobs in this field.

// Would you like more specific information about any of these courses or additional recommendations for self-study resources?`;
//     }

//     if (
//       input.includes("software") ||
//       input.includes("programming") ||
//       input.includes("developer")
//     ) {
//       return `Based on your interest in software development, here are some recommended courses:

// 1. **CS101: Introduction to Computer Science** - Foundational concepts in computing
// 2. **CS201: Data Structures and Algorithms** - Essential for coding interviews
// 3. **CS350: Software Engineering Principles** - Learn industry best practices
// 4. **WEB101: Full-Stack Web Development** - Practical skills for web applications

// These courses will help build a strong foundation for a software engineering career. Would you like more specialized recommendations in any particular area of software development?`;
//     }

//     if (input.includes("data") || input.includes("analytics")) {
//       return `For a career in data science or analytics, I recommend these courses:

// 1. **STAT101: Statistics for Data Analysis** - Build your foundation
// 2. **DS201: Introduction to Data Science** - Learn core concepts and tools
// 3. **ML301: Machine Learning Fundamentals** - Understand algorithms and models
// 4. **DS450: Big Data Technologies** - Work with large-scale data systems

// Would you like me to suggest more specialized courses based on particular industries you're interested in?`;
//     }

//     if (
//       input.includes("design") ||
//       input.includes("ux") ||
//       input.includes("ui")
//     ) {
//       return `For user experience and design, these courses would be valuable:

// 1. **DES101: Design Principles** - Fundamentals of good design
// 2. **UX201: User Experience Fundamentals** - Learn user-centered design methods
// 3. **UI301: User Interface Design** - Create effective and attractive interfaces
// 4. **DES401: Design Systems** - Build scalable design frameworks

// These will help you build the skills needed for a UX/UI design career. Would you like recommendations for portfolio-building courses as well?`;
//     }

//     // Generic response for broad or unclear inputs
//     return `Thank you for sharing your interests in "${userInput}". To provide the most relevant course recommendations, I'd like to know a bit more:

// 1. What specific career outcome are you hoping for?
// 2. Do you already have experience in this field?
// 3. Are you looking for introductory courses or more advanced specializations?

// This will help me suggest the best academic path for your goals.`;
//   };

//   // Define the main dashboard content
//   const DashboardContent = () => (
//     <div className="course-advisor-container">
//       <Sidebar
//         chats={transformedChats}
//         currentChatId={currentChatId}
//         onSelectChat={selectChat}
//         onCreateNewChat={createNewChat}
//         user={currentUser}
//         onLogout={handleLogout}
//         onCourseCatalogClick={handleCourseCatalogClick}
//         onMyCoursesClick={handleMyCoursesClick}
//       />

//       {/* Conditionally render either ChatInterface or CourseSearch based on activeView */}
//       {activeView === "chat" ? (
//         <ChatInterface chat={currentChat} onSendMessage={handleNewMessage} />
//       ) : (
//         <CourseSearch />
//       )}
//     </div>
//   );

//   return (
//     <Routes>
//       <Route
//         path="/login"
//         element={
//           isAuthenticated ? (
//             <Navigate to="/dashboard" replace />
//           ) : (
//             <LoginPageWithSlides onSignup={handleSignup} />
//           )
//         }
//       />
//       <Route
//         path="/dashboard"
//         element={
//           <ProtectedRoute>
//             <DashboardContent />
//           </ProtectedRoute>
//         }
//       />
//       <Route
//         path="/"
//         element={
//           <Navigate to={isAuthenticated ? "/dashboard" : "/login"} replace />
//         }
//       />
//       <Route
//         path="*"
//         element={
//           <Navigate to={isAuthenticated ? "/dashboard" : "/login"} replace />
//         }
//       />
//     </Routes>
//   );
// };

// export default AppContent;
import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { useAppSelector, useAppDispatch } from "../hooks/redux";
import LoginPageWithSlides from "./LoginPage";
import ProtectedRoute from "./ProtectedRoute";
import Dashboard from "./Dashboard";
import { setCredentials } from "../store/slices/authSlice";

setCredentials;
const AppContent: React.FC = () => {
  const dispatch = useAppDispatch();
  const { isAuthenticated } = useAppSelector((state) => state.auth);
  const handleSignup = async (
    username: string,
    password: string,
    firstName: string,
    lastName: string,
    email: string
  ): Promise<boolean> => {
    // In a real app, you would send this data to your backend
    return new Promise((resolve) => {
      // Simulate network delay
      setTimeout(() => {
        if (username && password && firstName && lastName && email) {
          // Generate a mock user ID (in a real app, this would come from the backend)
          const id = `user_${Math.floor(Math.random() * 10000)}`;

          // Create user data
          const userData = {
            id,
            username,
            firstName,
            lastName,
            email,
          };

          // Update Redux state
          dispatch(
            setCredentials({
              user: userData,
              token: `mock-token-${Date.now()}`,
            })
          );

          resolve(true);
        } else {
          resolve(false);
        }
      }, 800);
    });
  };
  return (
    <Routes>
      <Route
        path="/login"
        element={
          isAuthenticated ? (
            <Navigate to="/dashboard" replace />
          ) : (
            <LoginPageWithSlides onSignup={handleSignup} />
          )
        }
      />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/"
        element={
          <Navigate to={isAuthenticated ? "/dashboard" : "/login"} replace />
        }
      />
      <Route
        path="*"
        element={
          <Navigate to={isAuthenticated ? "/dashboard" : "/login"} replace />
        }
      />
    </Routes>
  );
};

export default AppContent;
