import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { faker } from "@faker-js/faker";
import { Chat, ChatMessage } from "../chatType";

// Create a tracking object for pending AI responses
export const pendingResponses = {
  // Maps chat IDs to pending status
  chatStatusMap: new Map<string, boolean>(),

  // Set a chat as pending
  setPending: (chatId: string, isPending: boolean) => {
    pendingResponses.chatStatusMap.set(chatId, isPending);
  },

  // Check if a chat has a pending response
  isPending: (chatId: string): boolean => {
    return pendingResponses.chatStatusMap.get(chatId) || false;
  },

  // Clear pending status
  clearPending: (chatId: string) => {
    pendingResponses.chatStatusMap.delete(chatId);
  },
};

export const chatApi = createApi({
  reducerPath: "chatApi",
  baseQuery: fetchBaseQuery({
    baseUrl: process.env.REACT_APP_API_BASE_URL || "http://192.168.0.139:3001",
    fetchFn: async (...args) => {
      // await pause(1000);
      return fetch(...args);
    },
    prepareHeaders: (headers, { getState }) => {
      // Get user from localStorage
      const user = localStorage.getItem("user");
      const parsedUser = user ? JSON.parse(user) : null;

      // If user exists, add user ID to the headers
      if (parsedUser && parsedUser.id) {
        headers.set("X-User-Id", parsedUser.id);
      }

      // Set content type for POST requests
      if (!headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
      }

      return headers;
    },
  }),
  tagTypes: ["Chats", "Messages"],
  endpoints: (builder) => ({
    // Call external AI API for generating responses
    getAIResponse: builder.mutation<
      { response: string; recommended_courses?: any[] },
      { user_id: string; query: string; session_id?: string | null }
    >({
      queryFn: async (request, _api, _options, fetchWithBQ) => {
        try {
          const chatId = request.session_id || "new-chat";

          // Set pending status to true
          pendingResponses.setPending(chatId, true);

          // Use the AI service endpoint
          const response = await fetch("http://192.168.0.126:8000/chat", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(request),
          });

          // Clear pending status
          pendingResponses.setPending(chatId, false);

          if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
          }

          const data = await response.json();
          return { data };
        } catch (error) {
          // Clear pending status on error
          if (request.session_id) {
            pendingResponses.clearPending(request.session_id);
          }

          // Return a friendly error for the frontend
          return {
            error: {
              status: "CUSTOM_ERROR",
              error: `Failed to get AI response: ${
                error instanceof Error ? error.message : String(error)
              }`,
            },
          };
        }
      },
    }),

    // Create a new chat for a user
    createChat: builder.mutation<Chat, { title?: string }>({
      async queryFn({ title }, _queryApi, _extraOptions, fetchWithBQ) {
        const user = localStorage.getItem("user");
        const parsedUser = user ? JSON.parse(user) : null;

        if (!parsedUser?.id) {
          throw new Error("User ID not found");
        }

        // Create a new chat with the updated schema
        const newChat: Chat = {
          id: faker.string.uuid(),
          userId: parsedUser.id,
          title: title || faker.lorem.words(3),
          created_at: new Date().toISOString(),
        };

        const newChatResponse = await fetchWithBQ({
          url: "/chats",
          method: "POST",
          body: newChat,
        });

        return { data: newChatResponse.data as Chat };
      },
      invalidatesTags: ["Chats"],
    }),

    // Add a message to an existing chat or create a new chat
    addChatMessage: builder.mutation<
      { chat: Chat; message: ChatMessage },
      {
        chatId?: string;
        message: string | object;
        role: "user" | "assistant";
        title?: string;
      }
    >({
      async queryFn(
        { chatId, message, role, title },
        _queryApi,
        _extraOptions,
        fetchWithBQ
      ) {
        const user = localStorage.getItem("user");
        const parsedUser = user ? JSON.parse(user) : null;

        if (!parsedUser?.id) {
          throw new Error("User ID not found");
        }

        // If no chatId provided, create a new chat
        let targetChatId = chatId;
        let chat: Chat;

        if (!targetChatId) {
          const newChat: Chat = {
            id: faker.string.uuid(),
            userId: parsedUser.id,
            title: title || faker.lorem.words(3),
            created_at: new Date().toISOString(),
          };

          const newChatResponse = await fetchWithBQ({
            url: "/chats",
            method: "POST",
            body: newChat,
          });

          chat = newChatResponse.data as Chat;
          targetChatId = chat.id;
        } else {
          // Get existing chat details
          const chatResponse = await fetchWithBQ({
            url: `/chats/${targetChatId}`,
            method: "GET",
          });

          chat = chatResponse.data as Chat;
        }

        // Prepare the new message with the updated schema
        const newMessage: ChatMessage = {
          id: faker.string.uuid(),
          chatId: targetChatId,
          role: role,
          message: message,
          created_at: new Date().toISOString(),
        };

        // Create the message
        const newMessageResponse = await fetchWithBQ({
          url: "/messages",
          method: "POST",
          body: newMessage,
        });

        return {
          data: {
            chat: chat,
            message: newMessageResponse.data as ChatMessage,
          },
        };
      },
      invalidatesTags: (result, error, { chatId }) => [
        "Messages",
        { type: "Chats", id: chatId },
      ],
    }),

    // Get user's chats
    getUserChats: builder.query<Chat[], void>({
      query: () => {
        const user = localStorage.getItem("user");
        const parsedUser = user ? JSON.parse(user) : null;

        if (!parsedUser?.id) {
          throw new Error("User ID not found");
        }

        return `/chats?userId=${parsedUser.id}`;
      },
      providesTags: ["Chats"],
    }),

    // Get messages for a specific chat
    getChatMessages: builder.query<ChatMessage[], string>({
      query: (chatId) => `/messages?chatId=${chatId}`,
      transformResponse: (response: ChatMessage[], _meta, chatId: string) => {
        // Sort messages by creation time
        const sortedMessages = response.sort(
          (a, b) =>
            new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        );

        return sortedMessages;
      },
      providesTags: (result, error, chatId) => [
        { type: "Messages", id: chatId },
      ],
    }),

    // Check if a chat has a pending AI response
    getChatPendingStatus: builder.query<boolean, string>({
      queryFn: (chatId) => {
        return { data: pendingResponses.isPending(chatId) };
      },
      keepUnusedDataFor: 0, // Don't cache this data
    }),

    // Get a specific chat
    getChatDetails: builder.query<Chat, string>({
      query: (chatId) => `/chats/${chatId}`,
      providesTags: (result, error, chatId) => [{ type: "Chats", id: chatId }],
    }),

    // Update a chat (for updating title, etc.)
    updateChat: builder.mutation<Chat, Partial<Chat> & { id: string }>({
      query: (chat) => ({
        url: `/chats/${chat.id}`,
        method: "PATCH",
        body: chat,
      }),
      invalidatesTags: (result, error, chat) => [
        { type: "Chats", id: chat.id },
        "Chats", // This will refetch the entire chats list
      ],
    }),

    // Process messages containing course recommendations
    processChatMessages: builder.query<ChatMessage[], string>({
      query: (chatId) => `/messages?chatId=${chatId}`,
      transformResponse: (response: ChatMessage[]) => {
        // Map and transform messages
        return response.map((msg) => {
          // Check if this message contains recommended courses
          if (msg.role === "assistant") {
            try {
              if (
                (typeof msg.message === "string" &&
                  msg.message.includes("recommended_courses")) ||
                (typeof msg.message === "object" &&
                  msg.message &&
                  "recommended_courses" in msg.message)
              ) {
                // Process message with recommendations
                // You can add custom logic here to extract or format recommendations
              }
            } catch (error) {
              console.error("Error processing message:", error);
            }
          }
          return msg;
        });
      },
      providesTags: (result, error, chatId) => [
        { type: "Messages", id: chatId },
      ],
    }),
  }),
});

// Export hooks for using the API
export const {
  useGetAIResponseMutation,
  useCreateChatMutation,
  useAddChatMessageMutation,
  useGetUserChatsQuery,
  useGetChatMessagesQuery,
  useGetChatDetailsQuery,
  useGetChatPendingStatusQuery,
  useUpdateChatMutation,
  useProcessChatMessagesQuery,
} = chatApi;
