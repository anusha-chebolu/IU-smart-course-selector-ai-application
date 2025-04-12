// components/UserProfile.tsx
import React from "react";

interface UserProfileProps {
  name: string;
  organization: string;
}

const UserProfile: React.FC<UserProfileProps> = ({ name, organization }) => {
  return (
    <div className="user-profile">
      <div className="profile-content">
        <div className="avatar">{name.charAt(0)}</div>
        <div className="user-info">
          <div className="user-name">{name}</div>
          <div className="user-org">{organization}</div>
        </div>
        <svg
          className="dropdown-icon"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </div>
    </div>
  );
};

export default UserProfile;
