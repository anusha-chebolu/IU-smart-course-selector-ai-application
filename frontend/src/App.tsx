// App.tsx
import React, { useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Provider } from "react-redux";
import { store } from "./store";
import { useAppDispatch } from "./hooks/redux";
import { checkAuthStatus } from "./store/slices/authSlice";
import AppContent from "./components/AppContent";
import "./CourseAdvisorInterface.css";
import "./index.css";

// Create a separate component for initializing auth status
const AppInitializer: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    // Check and restore authentication status on app load
    dispatch(checkAuthStatus());
  }, [dispatch]);

  return <>{children}</>;
};

// Separate the Provider from the app content to avoid React context issues
const App: React.FC = () => {
  return (
    <Provider store={store}>
      <Router>
        <AppInitializer>
          <Routes>
            <Route path="/*" element={<AppContent />} />
          </Routes>
        </AppInitializer>
      </Router>
    </Provider>
  );
};

export default App;
