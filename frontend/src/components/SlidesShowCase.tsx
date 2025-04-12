import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import User1 from "../assets/user1.png";
import User2 from "../assets/user2.png";
import User3 from "../assets/user3.png";

// Slide variants for animation
const slideVariants = {
  hidden: { opacity: 0, x: "100%" },
  visible: {
    opacity: 1,
    x: 0,
    transition: {
      type: "tween",
      duration: 0.5,
    },
  },
  exit: {
    opacity: 0,
    x: "-100%",
    transition: {
      type: "tween",
      duration: 0.5,
    },
  },
};

// Slides content
export const slides = [
  {
    userImage: User1,
    message: "What courses would you recommend for a Computer Science major?",
    content: (
      <div className="p-6">
        <h3 className="text-lg font-medium mb-3">Recommended CS Courses</h3>
        <div className="space-y-4">
          <div className="border-l-4 border-indigo-500 pl-4">
            <h4 className="font-medium">CS301: Data Structures & Algorithms</h4>
            <p className="text-sm text-gray-600">
              Foundation course covering essential algorithms and data
              structures
            </p>
            <div className="flex items-center mt-1">
              <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                High demand
              </span>
              <span className="text-xs ml-2">⭐ 4.8/5 (126 reviews)</span>
            </div>
          </div>
          <div className="border-l-4 border-indigo-500 pl-4">
            <h4 className="font-medium">
              CS410: Machine Learning Fundamentals
            </h4>
            <p className="text-sm text-gray-600">
              Introduction to ML algorithms and applications
            </p>
            <div className="flex items-center mt-1">
              <span className="text-xs bg-indigo-100 text-indigo-800 px-2 py-1 rounded">
                Popular
              </span>
              <span className="text-xs ml-2">⭐ 4.6/5 (98 reviews)</span>
            </div>
          </div>
          <div className="border-l-4 border-indigo-500 pl-4">
            <h4 className="font-medium">CS355: Database Systems</h4>
            <p className="text-sm text-gray-600">
              Design and implementation of database management systems
            </p>
            <div className="flex items-center mt-1">
              <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                Career essential
              </span>
              <span className="text-xs ml-2">⭐ 4.5/5 (112 reviews)</span>
            </div>
          </div>
        </div>
      </div>
    ),
    response: "Based on your major, here are my top course recommendations.",
  },
  {
    userImage: User2,
    message: "I'm struggling with calculus. What resources can help me?",
    content: (
      <div className="p-6">
        <h3 className="text-lg font-medium mb-3">Calculus Support Resources</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="flex items-center mb-2">
              <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center mr-2">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 text-indigo-600"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path d="M10.394 2.08a1 1 0 00-.788 0l-7 3a1 1 0 000 1.84L5.25 8.051a.999.999 0 01.356-.257l4-1.714a1 1 0 11.788 1.838L7.667 9.088l1.94.831a1 1 0 00.787 0l7-3a1 1 0 000-1.838l-7-3zM3.31 9.397L5 10.12v4.102a8.969 8.969 0 00-1.05-.174 1 1 0 01-.89-.89 11.115 11.115 0 01.25-3.762zM9.3 16.573A9.026 9.026 0 007 14.935v-3.957l1.818.78a3 3 0 002.364 0l5.508-2.361a11.026 11.026 0 01.25 3.762 1 1 0 01-.89.89 8.968 8.968 0 00-5.35 2.524 1 1 0 01-1.4 0zM6 18a1 1 0 001-1v-2.065a8.935 8.935 0 00-2-.712V17a1 1 0 001 1z" />
                </svg>
              </div>
              <h4 className="font-medium">Tutoring Center</h4>
            </div>
            <p className="text-sm text-gray-600">
              Free one-on-one tutoring sessions with math graduate students
            </p>
            <p className="text-xs text-indigo-600 mt-1">
              Mon-Fri, 9am-6pm in Math Building Room 302
            </p>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="flex items-center mb-2">
              <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center mr-2">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 text-indigo-600"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <h4 className="font-medium">Video Lessons</h4>
            </div>
            <p className="text-sm text-gray-600">
              Supplementary video lessons covering all calculus topics
            </p>
            <p className="text-xs text-indigo-600 mt-1">
              Available 24/7 on the university learning portal
            </p>
          </div>
        </div>
      </div>
    ),
    response:
      "Here are resources to help you with calculus - both on-campus and online.",
  },
  {
    userImage: User3,
    message:
      "Which electives should I take to prepare for a career in UX design?",
    content: (
      <div className="p-6 text-center">
        <h3 className="text-lg font-medium mb-3">UX Design Career Path</h3>
        <div className="space-y-4">
          <div className="border-l-4 border-indigo-500 pl-4 text-left">
            <h4 className="font-medium">Design Thinking Workshop</h4>
            <p className="text-sm text-gray-600">
              Learn user-centered design principles and problem-solving
              techniques
            </p>
            <div className="flex items-center mt-1">
              <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded mr-2">
                Recommended
              </span>
              <span className="text-xs">⭐ 4.7/5 (85 reviews)</span>
            </div>
          </div>
          <div className="border-l-4 border-indigo-500 pl-4 text-left">
            <h4 className="font-medium">Interactive Media Design</h4>
            <p className="text-sm text-gray-600">
              Advanced course in digital interaction and user experience
            </p>
            <div className="flex items-center mt-1">
              <span className="text-xs bg-indigo-100 text-indigo-800 px-2 py-1 rounded mr-2">
                Career Boost
              </span>
              <span className="text-xs">⭐ 4.6/5 (72 reviews)</span>
            </div>
          </div>
          <div className="border-l-4 border-indigo-500 pl-4 text-left">
            <h4 className="font-medium">Psychology of Design</h4>
            <p className="text-sm text-gray-600">
              Understanding user behavior and cognitive design principles
            </p>
            <div className="flex items-center mt-1">
              <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">
                Insight Course
              </span>
              <span className="text-xs">⭐ 4.8/5 (93 reviews)</span>
            </div>
          </div>
        </div>
      </div>
    ),
    response: "For a UX design career, here's your recommended course pathway.",
  },
];

interface SlidesShowcaseProps {
  className?: string;
}

const SlidesShowcase: React.FC<SlidesShowcaseProps> = ({ className = "" }) => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [autoplay, setAutoplay] = useState(true);

  // Slide auto-rotation
  useEffect(() => {
    let slideTimer: NodeJS.Timeout | null = null;

    if (autoplay) {
      slideTimer = setInterval(() => {
        setCurrentSlide((prev) => (prev + 1) % slides.length);
      }, 5000); // Change slide every 5 seconds
    }

    return () => {
      if (slideTimer) clearInterval(slideTimer);
    };
  }, [autoplay]);

  return (
    <div className={`relative w-full h-full overflow-hidden ${className}`}>
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-50 via-white to-blue-100 opacity-70"></div>

      <div className="relative h-full flex flex-col">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentSlide}
            variants={slideVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="flex-grow flex items-center justify-center px-4"
          >
            <div className="w-full max-w-md mx-auto text-center flex flex-col items-center">
              <div className="mb-4 flex justify-center items-center">
                <div className="w-10 h-10 rounded-full bg-gray-300 overflow-hidden mr-3 flex-shrink-0">
                  <img
                    src={slides[currentSlide].userImage}
                    alt="User"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="bg-gray-200 rounded-lg p-3 text-left flex-grow">
                  {slides[currentSlide].message}
                </div>
              </div>

              <div className="mt-2 mb-4 w-full">
                <div className="bg-white rounded-lg shadow-lg overflow-hidden">
                  <div className="relative">
                    <div className="p-3 bg-white rounded-t-lg text-center">
                      {slides[currentSlide].response}
                    </div>
                    <div className="absolute top-0 right-0 h-4 w-8 bg-white rounded-bl-lg"></div>
                  </div>
                  {slides[currentSlide].content}
                </div>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Slide Indicators */}
        <div className="absolute inset-x-0 bottom-4 flex justify-center items-center">
          <div className="flex space-x-2">
            {slides.map((_, index) => (
              <button
                key={index}
                onClick={() => {
                  setCurrentSlide(index);
                  setAutoplay(false);
                  setTimeout(() => setAutoplay(true), 10000);
                }}
                className={`w-2 h-2 rounded-full ${
                  index === currentSlide ? "bg-gray-500" : "bg-gray-300"
                }`}
                aria-label={`Go to slide ${index + 1}`}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SlidesShowcase;
