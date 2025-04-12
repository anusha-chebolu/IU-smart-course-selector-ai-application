import React from "react";

interface LogoProps {
  bgColor?: string;
  iconColor?: string;
  textColor?: string;
  className?: string;
}

const Logo: React.FC<LogoProps> = ({
  bgColor = "#4F46E5",
  iconColor = "white",
  textColor = "text-gray-900",
  className = "",
}) => {
  return (
    <div className={`logo-container flex items-center space-x-2 ${className}`}>
      <svg
        className="logo-icon w-6 h-6"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <rect width="24" height="24" rx="4" fill={bgColor} />
        <path
          d="M12 6V18M7 10L17 10M7 14L17 14"
          stroke={iconColor}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <h1 className={`logo-text font-bold text-xl ${textColor}`}>
        Course Advisor
      </h1>
    </div>
  );
};

export default Logo;
