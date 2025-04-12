# Course Advisor Application

A React application for personalized course recommendations and academic guidance.

## Features

- User authentication (signup and login)
- Personalized course recommendations
- Modern, responsive UI with smooth transitions
- Course information showcase
- Dashboard for authenticated users

## Tech Stack

- React with TypeScript
- Redux Toolkit with RTK Query for state management and API calls
- React Router for navigation
- Tailwind CSS for styling
- JSON Server with JWT authentication for backend mocking

## Setup and Installation

### Prerequisites

- Node.js (version 14 or higher)
- npm or yarn

### Installation Steps

1. Clone the repository:

```bash
git clone <repository-url>
cd course-advisor
```

2. Install dependencies:

```bash
npm install
# or
yarn install
```

3. Install additional dependencies for the JSON Server:

```bash
npm install -g json-server
npm install jsonwebtoken bcryptjs
```

### Running the Application

1. Start the JSON Server (mock backend):

```bash

```

This will start the server at http://localhost:3001

2. In a separate terminal, start the React application:

```bash
npm start
# or
yarn start
```

This will start the application at http://localhost:3000

## Project Structure

```
src/
├── components/            # React components
│   ├── Dashboard.tsx
│   ├── LoginForm.tsx
│   ├── LoginPageWithSlides.tsx
│   ├── Logo.tsx
│   ├── ProtectedRoute.tsx
│   ├── SignupForm.tsx
│   └── SlidesShowcase.tsx
├── hooks/                 # Custom hooks
│   └── redux.ts
├── services/              # API services
│   └── api.ts
├── store/                 # Redux store setup
│   ├── index.ts
│   └── slices/
│       └── authSlice.ts
├── types/                 # TypeScript type definitions
├── App.tsx                # Main application component
├── index.tsx              # Application entry point
└── ...
```

## API Endpoints

The JSON Server exposes these endpoints:

- **POST /login** - Authenticate a user
- **POST /users** - Create a new user (signup)
- **GET /profile** - Get the current user's profile (protected)
- **GET /courses** - Get course listings

## Default User

You can log in with the following credentials to test the application:

- Username: demo
- Password: password123

## Development

### Adding New Features

1. Create new component files in the `components` directory
2. Add API endpoints in `services/api.ts`
3. Add Redux state management in `store/slices` if needed
4. Update routes in `App.tsx`

### Building for Production

```bash
npm run build
# or
yarn build
```

This will create an optimized production build in the `build` directory.

## License

[MIT](LICENSE)
