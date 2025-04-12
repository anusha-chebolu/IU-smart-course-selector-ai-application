import React, { useState } from "react";
import Logo from "./Logo";

interface SignupModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLogin: (username: string, password: string) => Promise<boolean> | boolean;
  onSignup: (
    username: string,
    password: string,
    firstName: string,
    lastName: string,
    email: string
  ) => Promise<boolean> | boolean;
}

const SignupModal: React.FC<SignupModalProps> = ({
  isOpen,
  onClose,
  onLogin,
  onSignup,
}) => {
  const [isLoginMode, setIsLoginMode] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      if (isLoginMode) {
        // Login logic
        if (!username || !password) {
          setError("Please enter username and password");
          setIsLoading(false);
          return;
        }

        const success = await onLogin(username, password);
        if (success) {
          onClose();
        } else {
          setError("Invalid username or password");
        }
      } else {
        // Signup logic
        if (!username) {
          setError("Username is required");
          setIsLoading(false);
          return;
        }

        if (!firstName || !lastName) {
          setError("First name and last name are required");
          setIsLoading(false);
          return;
        }

        if (!email) {
          setError("Email is required");
          setIsLoading(false);
          return;
        }

        // Basic email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
          setError("Please enter a valid email address");
          setIsLoading(false);
          return;
        }

        if (password.length < 8) {
          setError("Password must be at least 8 characters long");
          setIsLoading(false);
          return;
        }

        if (password !== confirmPassword) {
          setError("Passwords do not match");
          setIsLoading(false);
          return;
        }

        const success = await onSignup(
          username,
          password,
          firstName,
          lastName,
          email
        );
        if (success) {
          onClose();
        } else {
          setError("Signup failed. Please try again.");
        }
      }
    } catch (err) {
      console.error("Authentication error:", err);
      setError("An unexpected error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  // Reset form when switching modes
  const switchMode = () => {
    setIsLoginMode(!isLoginMode);
    setUsername("");
    setPassword("");
    setConfirmPassword("");
    setFirstName("");
    setLastName("");
    setEmail("");
    setError("");
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4 p-8">
        <div className="mb-6">
          <Logo />
        </div>

        <div className="mb-6">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            {isLoginMode ? "Welcome Back" : "Create an Account"}
          </h2>
          <p className="text-gray-600">
            {isLoginMode
              ? "Sign in to continue your academic journey"
              : "Start your academic journey with Course Advisor"}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="auth-username"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Username
            </label>
            <input
              id="auth-username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder={
                isLoginMode ? "Enter your username" : "Choose a username"
              }
            />
          </div>

          {!isLoginMode && (
            <>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label
                    htmlFor="auth-first-name"
                    className="block text-sm font-medium text-gray-700 mb-2"
                  >
                    First Name
                  </label>
                  <input
                    id="auth-first-name"
                    type="text"
                    value={firstName}
                    onChange={(e) => setFirstName(e.target.value)}
                    required
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    placeholder="First name"
                  />
                </div>
                <div>
                  <label
                    htmlFor="auth-last-name"
                    className="block text-sm font-medium text-gray-700 mb-2"
                  >
                    Last Name
                  </label>
                  <input
                    id="auth-last-name"
                    type="text"
                    value={lastName}
                    onChange={(e) => setLastName(e.target.value)}
                    required
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    placeholder="Last name"
                  />
                </div>
              </div>

              <div>
                <label
                  htmlFor="auth-email"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Email
                </label>
                <input
                  id="auth-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="your.email@example.com"
                />
              </div>
            </>
          )}

          <div>
            <label
              htmlFor="auth-password"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Password
            </label>
            <input
              id="auth-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder={
                isLoginMode ? "Enter your password" : "Create a password"
              }
            />
          </div>

          {!isLoginMode && (
            <div>
              <label
                htmlFor="auth-confirm-password"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Confirm Password
              </label>
              <input
                id="auth-confirm-password"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                placeholder="Confirm your password"
              />
            </div>
          )}

          {error && (
            <div className="text-red-600 text-sm text-center">{error}</div>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
          >
            {isLoading
              ? isLoginMode
                ? "Logging in..."
                : "Creating Account..."
              : isLoginMode
              ? "Sign In"
              : "Sign Up"}
          </button>
        </form>

        <div className="mt-4 text-center text-sm text-gray-600">
          {isLoginMode
            ? "Don't have an account? "
            : "Already have an account? "}
          <button
            onClick={switchMode}
            className="text-indigo-600 hover:text-indigo-500 font-medium"
          >
            {isLoginMode ? "Sign up" : "Log in"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default SignupModal;
