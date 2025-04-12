import { configureStore } from "@reduxjs/toolkit";
import { setupListeners } from "@reduxjs/toolkit/query";
import { api } from "../services/api";
import { coursesApi, courseTrendsApi } from "../services/courses";
import authReducer from "./slices/authSlice";
import { chatApi } from "../services/chat_api";
import { myCourseApi } from "../services/mycourses_api";

export const store = configureStore({
  reducer: {
    auth: authReducer,
    [api.reducerPath]: api.reducer,
    [chatApi.reducerPath]: chatApi.reducer,
    [coursesApi.reducerPath]: coursesApi.reducer,
    [myCourseApi.reducerPath]: myCourseApi.reducer,
    [courseTrendsApi.reducerPath]: courseTrendsApi.reducer,
    // Add coursesApi reducer
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware()
      .concat(api.middleware)
      .concat(chatApi.middleware)
      .concat(coursesApi.middleware)
      .concat(myCourseApi.middleware)
      .concat(courseTrendsApi.middleware),
  // Add coursesApi middleware
});

// Optional but recommended for refetchOnFocus/refetchOnReconnect behaviors
setupListeners(store.dispatch);

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
