import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { RootState } from "../store";
import { setCredentials, clearCredentials } from "../store/slices/authSlice";
import { User, ChatMessage } from "../types";

export interface LoginRequest {
  username: string;
  password: string;
}

export interface SignupRequest {
  username: string;
  password: string;
  firstName: string;
  lastName: string;
  email: string;
}

export interface AuthResponse {
  user: User;
  token: string;
}

// External API response type
export interface ExternalAuthResponse {
  user_id: string;
  first_name: string;
  last_name: string;
  email: string;
  password: string;
}

export const api = createApi({
  reducerPath: "api",
  baseQuery: fetchBaseQuery({
    baseUrl: process.env.REACT_APP_API_BASE_URL || "http://192.168.0.139:3001",
    prepareHeaders: (headers, { getState }) => {
      // Get token from Redux store or localStorage
      const token =
        (getState() as RootState).auth?.token || localStorage.getItem("token");

      // If token exists, add it to the headers
      if (token) {
        headers.set("Authorization", `Bearer ${token}`);
      }

      // Set content type for POST requests
      if (!headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
      }

      return headers;
    },
  }),
  tagTypes: ["User", "Profile"],
  endpoints: (builder) => ({
    // AUTHENTICATION ENDPOINTS

    // Login endpoint - Now with integrated user check and creation
    login: builder.mutation<AuthResponse, LoginRequest>({
      async queryFn(credentials, _queryApi, _extraOptions, fetchWithBQ) {
        try {
          // Step 1: Authenticate with external API
          const externalLoginResult = await fetchWithBQ({
            url: "http://192.168.0.126:8000/login",
            method: "POST",
            body: {
              user_id: credentials.username,
              password: credentials.password,
            },
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
            },
          });

          if (externalLoginResult.error) {
            return { error: externalLoginResult.error };
          }

          const externalResponse =
            externalLoginResult.data as ExternalAuthResponse;

          // Transform the external API response to our internal User format
          const user: User = {
            id: externalResponse.user_id,
            username: credentials.username,
            firstName: externalResponse.first_name,
            lastName: externalResponse.last_name,
            email: externalResponse.email,
            createdAt: new Date().toISOString(),
            password: credentials.password,
          };

          // Generate a token
          const token = `auth-token-${Date.now()}`;

          // Step 2: Check if user exists in our local DB
          const checkUserResult = await fetchWithBQ({
            url: "/users",
            method: "GET",
            params: { id: user.id },
          });

          const localUsers = (checkUserResult.data as User[]) || [];

          // Step 3: If user doesn't exist in our DB, create it
          if (!localUsers || localUsers.length === 0) {
            // User doesn't exist, create it
            await fetchWithBQ({
              url: "/users",
              method: "POST",
              body: user,
            });
          }

          // Return successful auth response
          return {
            data: {
              user,
              token,
            },
          };
        } catch (error) {
          return {
            error: {
              status: "CUSTOM_ERROR",
              error: "Error during login process: " + error,
            },
          };
        }
      },
      async onQueryStarted(_, { queryFulfilled, dispatch }) {
        try {
          const { data } = await queryFulfilled;

          // Save token to localStorage
          localStorage.setItem("token", data.token);

          // Save user to localStorage
          localStorage.setItem("user", JSON.stringify(data.user));

          // Update Redux store
          dispatch(
            setCredentials({
              user: data.user,
              token: data.token,
            })
          );
        } catch (error) {
          console.error("Login failed:", error);
          throw error;
        }
      },
    }),

    // Fallback to local users if external API fails
    getLocalUser: builder.query<User[], LoginRequest>({
      query: (credentials) => ({
        url: "/users",
        method: "GET",
        params: credentials,
      }),
    }),

    // Signup endpoint - With integrated user creation
    signup: builder.mutation<AuthResponse, SignupRequest>({
      async queryFn(userData, _queryApi, _extraOptions, fetchWithBQ) {
        try {
          // Step 1: Register with external API
          const externalSignupResult = await fetchWithBQ({
            url: "http://192.168.0.126:8000/signup",
            method: "POST",
            body: {
              user_id: userData.username,
              first_name: userData.firstName,
              last_name: userData.lastName,
              email: userData.email,
              password: userData.password,
            },
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
            },
          });

          if (externalSignupResult.error) {
            return { error: externalSignupResult.error };
          }

          const externalResponse =
            externalSignupResult.data as ExternalAuthResponse;

          // Transform the external API response to our internal User format
          const user: User = {
            id: externalResponse.user_id,
            username: userData.username,
            firstName: externalResponse.first_name,
            lastName: externalResponse.last_name,
            email: externalResponse.email,
            createdAt: new Date().toISOString(),
          };

          // Generate a token
          const token = `auth-token-${Date.now()}`;

          // Step 2: Create user in our local DB
          await fetchWithBQ({
            url: "/users",
            method: "POST",
            body: user,
          });

          // Return successful auth response
          return {
            data: {
              user,
              token,
            },
          };
        } catch (error) {
          return {
            error: {
              status: "CUSTOM_ERROR",
              error: "Error during signup process: " + error,
            },
          };
        }
      },
      async onQueryStarted(_, { queryFulfilled, dispatch }) {
        try {
          const { data } = await queryFulfilled;

          // Save token to localStorage
          localStorage.setItem("token", data.token);

          // Save user to localStorage
          localStorage.setItem("user", JSON.stringify(data.user));

          // Update Redux store
          dispatch(
            setCredentials({
              user: data.user,
              token: data.token,
            })
          );
        } catch (error) {
          console.error("Signup failed:", error);
          throw error;
        }
      },
      invalidatesTags: ["User", "Profile"],
    }),

    // Logout endpoint
    logout: builder.mutation<void, void>({
      queryFn: () => {
        // Client-side logout
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        return { data: undefined };
      },
      async onQueryStarted(_, { dispatch }) {
        // Clear authentication state in Redux
        dispatch(clearCredentials());
      },
      invalidatesTags: ["User", "Profile"],
    }),

    // Get current user profile
    getCurrentUser: builder.query<User, void>({
      query: () => `/users/current`,
      providesTags: ["Profile"],
    }),

    // Update user profile
    updateProfile: builder.mutation<User, Partial<User>>({
      query: (userData) => ({
        url: `/users/${userData.id}`,
        method: "PATCH",
        body: userData,
      }),
      invalidatesTags: ["Profile"],
    }),
  }),
});

// Export hooks for using the API
export const {
  useLoginMutation,
  useLazyGetLocalUserQuery,
  useSignupMutation,
  useLogoutMutation,
  useGetCurrentUserQuery,
  useUpdateProfileMutation,
} = api;

// Helper function to check if user is authenticated
export const isAuthenticated = () => {
  const token = localStorage.getItem("token");
  return !!token;
};
