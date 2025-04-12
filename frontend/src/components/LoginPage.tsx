// components/LoginPage.tsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Logo from "./Logo";
import SlidesShowcase from "./SlidesShowCase";
import LoginForm from "./LoginForm";
import SignupForm from "./SignupForm";
import { useAppSelector, useAppDispatch } from "../hooks/redux";
import { startLoading, authError } from "../store/slices/authSlice";

interface LoginPageWithSlidesProps {
  // onLogin: (username: string, password: string) => Promise<boolean>;
  onSignup: (
    username: string,
    password: string,
    firstName: string,
    lastName: string,
    email: string
  ) => Promise<boolean>;
}

const LoginPageWithSlides: React.FC<LoginPageWithSlidesProps> = ({
  // onLogin,
  onSignup,
}) => {
  const [isLoginView, setIsLoginView] = useState(true);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  // Get authentication state from Redux
  const { isAuthenticated, isLoading } = useAppSelector((state) => state.auth);

  // Handle navigation on auth change
  useEffect(() => {
    if (isAuthenticated) {
      navigate("/dashboard");
    }
  }, [isAuthenticated, navigate]);

  // Enhanced login handler with Redux actions
  // const handleLogin = async (
  //   username: string,
  //   password: string
  // ): Promise<boolean> => {
  //   dispatch(startLoading());

  //   try {
  //     return await onLogin(username, password);
  //   } catch (error) {
  //     dispatch(
  //       authError(error instanceof Error ? error.message : "Unknown error")
  //     );
  //     return false;
  //   }
  // };

  // Enhanced signup handler with Redux actions
  const handleSignup = async (
    username: string,
    password: string,
    firstName: string,
    lastName: string,
    email: string
  ): Promise<boolean> => {
    dispatch(startLoading());

    try {
      return await onSignup(username, password, firstName, lastName, email);
    } catch (error) {
      dispatch(
        authError(error instanceof Error ? error.message : "Unknown error")
      );
      return false;
    }
  };

  const toggleView = () => {
    if (isLoading) return; // Prevent toggling while authentication is in progress

    setIsTransitioning(true);

    // Apply transition when switching views
    setTimeout(() => {
      setIsLoginView(!isLoginView);
      setTimeout(() => {
        setIsTransitioning(false);
      }, 50);
    }, 300);
  };

  return (
    <div className="min-h-screen flex flex-col md:flex-row">
      {/* Left Section - Auth Forms */}
      <div className="flex-1 p-8 flex flex-col overflow-y-auto">
        <div className="mb-6">
          <Logo />
        </div>

        <div
          className={`flex-grow flex flex-col justify-center max-w-md mx-auto w-full space-y-6 transition-opacity duration-300 ease-in-out ${
            isTransitioning
              ? "opacity-0 transform translate-y-4"
              : "opacity-100 transform translate-y-0"
          }`}
        >
          {isLoginView ? (
            <LoginForm onSwitchToSignup={toggleView} />
          ) : (
            <SignupForm onSignup={handleSignup} onSwitchToLogin={toggleView} />
          )}
        </div>
      </div>

      {/* Right Section - Slides Showcase */}
      <div className="hidden md:block md:flex-1 bg-indigo-900 relative overflow-hidden">
        <div className="absolute inset-0 z-10"></div>
        <SlidesShowcase />
      </div>
    </div>
  );
};

export default LoginPageWithSlides;
