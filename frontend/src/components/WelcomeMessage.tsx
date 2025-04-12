import React from "react";

interface WelcomeMessageProps {
  userName: string;
}

const WelcomeMessage: React.FC<WelcomeMessageProps> = ({ userName }) => {
  try {
    return (
      <div className="welcome-message" data-name="welcome-message">
        <h1 className="text-4xl mb-4" data-name="welcome-title">
          Hello, I'm Claude.
        </h1>
        <p className="text-xl mb-6" data-name="welcome-subtitle">
          I'm a next generation AI assistant built for work and trained to be
          safe, accurate, and secure.
        </p>
        <p className="text-xl" data-name="welcome-greeting">
          I'd love for us to get to know each other a bit better.
        </p>
        <div
          className="mt-8 p-6 bg-gray-800 rounded-lg"
          data-name="name-input-section"
        >
          <p className="text-gray-400 mb-2">Nice to meet you, I'm...</p>
          <input
            type="text"
            value={userName}
            readOnly
            className="bg-transparent text-white text-xl border-none outline-none"
          />
          <p className="text-gray-400 mt-4">You can always change this later</p>
        </div>
      </div>
    );
  } catch (error) {
    console.error("WelcomeMessage error:", error);
    return null;
  }
};

export default WelcomeMessage;
