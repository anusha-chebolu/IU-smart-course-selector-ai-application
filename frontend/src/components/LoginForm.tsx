import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAppDispatch } from "../hooks/redux";
import { useLoginMutation, useLazyGetLocalUserQuery } from "../services/api";
import { setCredentials } from "../store/slices/authSlice";

interface LoginFormProps {
  onSwitchToSignup: () => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ onSwitchToSignup }) => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [rememberMe, setRememberMe] = useState(false);

  // New login mutation hook for external API
  const [login, { isLoading: isExternalLoading }] = useLoginMutation();

  // Fallback to local users if external API fails
  const [getLocalUser, { isLoading: isLocalLoading }] =
    useLazyGetLocalUserQuery();

  const dispatch = useAppDispatch();
  const navigate = useNavigate();

  // Combined loading state
  const isLoading = isExternalLoading || isLocalLoading;

  // Check for previously saved "remember me" preference
  useEffect(() => {
    const rememberedPreference = localStorage.getItem("rememberMe") === "true";
    setRememberMe(rememberedPreference);
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Basic validation
    if (!username.trim()) {
      setError("Username is required");
      return;
    }

    if (!password) {
      setError("Password is required");
      return;
    }

    try {
      // First try external API
      const result = await login({
        username,
        password,
      }).unwrap();

      // Success with external API
      // The auth token and user data are already stored by the API middleware

      // Save remember me preference
      if (rememberMe) {
        localStorage.setItem("rememberMe", "true");
      } else {
        localStorage.removeItem("rememberMe");
      }

      // Navigate to dashboard
      navigate("/dashboard");
    } catch (externalError) {
      console.error("External login failed, trying local:", externalError);

      try {
        // Fallback to local users if external API fails
        const { data } = await getLocalUser({ username, password });

        // Check if data is an empty array
        if (!data || data.length === 0) {
          setError("Invalid username or password");
          return;
        }

        // Assuming the first user in the array is the correct one
        const user = data[0];

        dispatch(
          setCredentials({
            user,
            token: `mock-token-${Date.now()}`, // Generate mock token
          })
        );

        // Save remember me preference
        if (rememberMe) {
          localStorage.setItem("rememberMe", "true");
        } else {
          localStorage.removeItem("rememberMe");
        }

        // Navigate to dashboard
        navigate("/dashboard");
      } catch (localError) {
        setError("Invalid username or password");
        console.error("Local login failed:", localError);
      }
    }
  };

  return (
    <div className="w-full">
      <div>
        <h1 className="text-4xl font-bold text-gray-900 mb-3">
          Your courses,
          <br />
          your future
        </h1>
        <p className="text-gray-600">
          Personalized academic guidance to help you succeed.
        </p>
      </div>

      <form
        onSubmit={handleLogin}
        className="space-y-4 mt-6"
        autoComplete="off"
      >
        <div>
          <label
            htmlFor="username"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Username
          </label>
          <input
            id="username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            placeholder="Enter your username"
            disabled={isLoading}
            autoComplete="new-username"
          />
        </div>

        <div>
          <label
            htmlFor="password"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Password
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            placeholder="Enter your password"
            disabled={isLoading}
            autoComplete="new-password"
          />
        </div>

        {error && (
          <div className="text-red-600 text-sm p-2 bg-red-50 border border-red-100 rounded text-center">
            {error}
          </div>
        )}

        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <input
              id="remember-me"
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
              className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
              disabled={isLoading}
            />
            <label
              htmlFor="remember-me"
              className="ml-2 block text-sm text-gray-700"
            >
              Remember me
            </label>
          </div>
          <div>
            <button
              type="button"
              className="text-sm text-indigo-600 hover:text-indigo-500"
              disabled={isLoading}
            >
              Forgot password?
            </button>
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-3 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors shadow-md disabled:bg-indigo-400 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <div className="flex items-center justify-center">
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Signing in...
            </div>
          ) : (
            "Sign In"
          )}
        </button>
      </form>

      <div className="text-center text-sm text-gray-600 mt-6">
        Don't have an account?{" "}
        <button
          onClick={onSwitchToSignup}
          className="text-indigo-600 hover:text-indigo-500 font-medium transition-colors"
          disabled={isLoading}
        >
          Sign up
        </button>
      </div>

      <div className="text-center text-xs text-gray-500 mt-4">
        By continuing, you agree to our Terms of Service and Privacy Policy.
      </div>
    </div>
  );
};

export default LoginForm;
