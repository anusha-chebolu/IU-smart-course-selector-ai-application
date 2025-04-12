const jsonServer = require("json-server");
const server = jsonServer.create();
const path = require("path");
const router = jsonServer.router(path.resolve(__dirname, "db.json"));
const middlewares = jsonServer.defaults();
const jwt = require("jsonwebtoken");
const bcrypt = require("bcryptjs");

// Set the JWT secret key (use environment variable in production)
const JWT_SECRET =
  process.env.JWT_SECRET || "your-secret-key-change-in-production";

// Use default middlewares (cors, static, etc)
server.use(middlewares);

// Parse request body as JSON
server.use(jsonServer.bodyParser);

// Authorization middleware
const authMiddleware = (req, res, next) => {
  // Skip authorization for login and signup endpoints
  if (
    req.path === "/auth/login" ||
    (req.path === "/users" && req.method === "POST")
  ) {
    return next();
  }

  // Check for auth header for protected routes
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith("Bearer ")) {
    const token = authHeader.split(" ")[1];
    try {
      const decoded = jwt.verify(token, JWT_SECRET);
      req.user = decoded;
      next();
    } catch (err) {
      res.status(401).json({ error: "Invalid or expired token" });
    }
  } else {
    // Allow access to public routes
    if (req.path === "/courses" && req.method === "GET") {
      return next();
    }

    res.status(401).json({ error: "Authentication required" });
  }
};

// Use authorization middleware
server.use(authMiddleware);

// Login endpoint
server.post("/auth/login", (req, res) => {
  const { username, password } = req.body;

  // Validate input
  if (!username || !password) {
    return res
      .status(400)
      .json({ error: "Username and password are required" });
  }

  // Get users from db
  const db = router.db.getState();
  const user = db.users.find((u) => u.username === username);

  // Check user and password
  if (!user) {
    return res.status(401).json({ error: "Invalid username or password" });
  }

  // In a real app, use bcrypt to compare passwords
  const isPasswordValid = user.password === password;
  // const isPasswordValid = bcrypt.compareSync(password, user.password);

  if (!isPasswordValid) {
    return res.status(401).json({ error: "Invalid username or password" });
  }

  // Create JWT token
  const token = jwt.sign(
    {
      id: user.id,
      username: user.username,
      email: user.email,
    },
    JWT_SECRET,
    { expiresIn: "24h" }
  );

  // Remove sensitive information
  const { password: _, ...userWithoutPassword } = user;

  res.json({
    user: userWithoutPassword,
    token,
  });
});

// Handle user registration
server.post("/users", (req, res) => {
  const { username, password, email, firstName, lastName } = req.body;

  // Validate required fields
  if (!username || !password || !email || !firstName || !lastName) {
    return res.status(400).json({ error: "All fields are required" });
  }

  // Get current database state
  const db = router.db.getState();

  // Check if username or email already exists
  const userExists = db.users.find(
    (u) => u.username === username || u.email === email
  );

  if (userExists) {
    return res.status(409).json({ error: "Username or email already exists" });
  }

  // Prepare new user data
  const newUser = {
    id: Date.now().toString(),
    username,
    email,
    firstName,
    lastName,
    // Hash password in real app
    password, // In production: bcrypt.hashSync(password, 10)
    createdAt: new Date().toISOString(),
  };

  // Add user to database
  db.users.push(newUser);
  router.db.write();

  // Create JWT token
  const token = jwt.sign(
    {
      id: newUser.id,
      username: newUser.username,
      email: newUser.email,
    },
    JWT_SECRET,
    { expiresIn: "24h" }
  );

  // Remove password before sending response
  const { password: _, ...userWithoutPassword } = newUser;

  res.status(201).json({
    user: userWithoutPassword,
    token,
  });
});

// Get current user profile
server.get("/auth/profile", (req, res) => {
  const db = router.db.getState();
  const user = db.users.find((u) => u.id === req.user.id);

  if (!user) {
    return res.status(404).json({ error: "User not found" });
  }

  // Remove password from response
  const { password, ...userWithoutPassword } = user;

  res.json(userWithoutPassword);
});

// Use default router for other routes
server.use(router);

// Start the server
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`JSON Server with auth is running on port ${PORT}`);
});
