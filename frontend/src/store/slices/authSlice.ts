import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { User } from "../../types";

// Define the state structure
interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

// Utility function to safely get stored user
const getStoredUser = (): User | null => {
  try {
    const userStr = localStorage.getItem("user");
    return userStr ? JSON.parse(userStr) : null;
  } catch (e) {
    console.error("Error parsing stored user:", e);
    localStorage.removeItem("user");
    return null;
  }
};

// Utility function to validate token (basic implementation)
const isValidToken = (token: string | null): boolean => {
  // For now, just check if token exists
  return !!token;
};

// Initialize state
const initialState: AuthState = {
  user: getStoredUser(),
  token: localStorage.getItem("token"),
  isAuthenticated: isValidToken(localStorage.getItem("token")),
  isLoading: false,
  error: null,
};

// Create the auth slice
const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    // Start loading state
    startLoading: (state) => {
      state.isLoading = true;
      state.error = null;
    },

    // Authentication error
    authError: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;

      // Clear credentials on auth error
      state.user = null;
      state.token = null;
      state.isAuthenticated = false;
      localStorage.removeItem("token");
      localStorage.removeItem("user");
    },

    // Set user and token after login/signup
    setCredentials: (
      state,
      action: PayloadAction<{ user: User; token: string }>
    ) => {
      const { user, token } = action.payload;

      state.user = user;
      state.token = token;
      state.isAuthenticated = true;
      state.isLoading = false;
      state.error = null;

      // Store in localStorage for persistence
      localStorage.setItem("token", token);
      localStorage.setItem("user", JSON.stringify(user));
    },

    // Clear credentials on logout
    clearCredentials: (state) => {
      state.user = null;
      state.token = null;
      state.isAuthenticated = false;

      // Remove from localStorage
      localStorage.removeItem("token");
      localStorage.removeItem("user");
    },

    // Update user information
    updateUser: (state, action: PayloadAction<Partial<User>>) => {
      if (state.user) {
        // Merge existing user with new data
        state.user = {
          ...state.user,
          ...action.payload,
        };

        // Update localStorage
        localStorage.setItem("user", JSON.stringify(state.user));
      }
    },

    // Check and refresh authentication status
    checkAuthStatus: (state) => {
      const token = localStorage.getItem("token");
      const user = getStoredUser();

      // Update auth state based on stored token and user
      state.token = token;
      state.user = user;
      state.isAuthenticated = !!token && !!user;
    },
  },
});

// Export actions and reducer
export const {
  startLoading,
  authError,
  setCredentials,
  clearCredentials,
  updateUser,
  checkAuthStatus,
} = authSlice.actions;

export default authSlice.reducer;

// Selector for getting current auth state
export const selectAuth = (state: { auth: AuthState }) => state.auth;
