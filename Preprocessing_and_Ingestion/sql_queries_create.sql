-- First, ensure the schema exists
CREATE SCHEMA IF NOT EXISTS iu_catalog;

-- Install the pgvector extension (if you have permissions)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create student_profile table with correct reference
CREATE TABLE iu_catalog.student_profile (
  user_id VARCHAR(20) PRIMARY KEY,
  degree_type VARCHAR(50) NOT NULL,
  major VARCHAR(100) NOT NULL,
  enrollment_type VARCHAR(20) CHECK (enrollment_type IN ('Full-time', 'Part-time')),
  gpa NUMERIC(3,2) DEFAULT 0.00 CHECK (gpa BETWEEN 0.00 AND 4.00),
  upcoming_semester VARCHAR(10) CHECK (upcoming_semester IN ('fall', 'spring', 'winter', 'summer')) NOT NULL,
  total_credits INTEGER NOT NULL DEFAULT 0,
  completed_credits INTEGER NOT NULL DEFAULT 0,
  remaining_credits INTEGER NOT NULL DEFAULT 0,
  time_availability JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create login table first
CREATE TABLE iu_catalog.login (
  user_id VARCHAR(20) PRIMARY KEY REFERENCES iu_catalog.student_profile(user_id) ON DELETE CASCADE,
  first_name VARCHAR(100) NOT NULL,
  last_name VARCHAR(100) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_login TIMESTAMP
);



-- Create course_details table before tables that reference it
-- Removed availability column as requested
CREATE TABLE iu_catalog.course_details (
  id SERIAL PRIMARY KEY,
  course_id VARCHAR(20) UNIQUE NOT NULL,
  course_name VARCHAR(255) NOT NULL,
  department VARCHAR(100) NOT NULL,
  min_credits INTEGER NOT NULL,
  max_credits INTEGER NOT NULL,
  prerequisites VARCHAR[] DEFAULT '{}',
  offered_semester VARCHAR(100) NOT NULL,
  course_title VARCHAR(255) NOT NULL,
  course_description TEXT,
  course_details JSONB,
  embedding VECTOR(384),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create completed_courses with correct references
CREATE TABLE iu_catalog.completed_courses (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(20) REFERENCES iu_catalog.student_profile(user_id) ON DELETE CASCADE,
  course_id VARCHAR(20) NOT NULL,
  semester VARCHAR(20) NOT NULL,
  credits INTEGER NOT NULL,
  year INTEGER NOT NULL,
  gpa FLOAT,
  completed_date TIMESTAMP,
  UNIQUE(user_id, course_id)
);

-- Create my_course_list table (NEW)
CREATE TABLE iu_catalog.my_course_list (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(20) REFERENCES iu_catalog.student_profile(user_id) ON DELETE CASCADE,
  course_id VARCHAR(20) REFERENCES iu_catalog.course_details(course_id) ON DELETE CASCADE,
  semester VARCHAR(20),
  year INTEGER,
  priority INTEGER DEFAULT 0, -- Higher number means higher priority
  notes TEXT,
  recommendation_id INTEGER, -- To link back to the recommendation that suggested this course
  added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, course_id, semester, year)
);

-- Create course_trends with correct references
CREATE TABLE iu_catalog.course_trends (
  id SERIAL PRIMARY KEY,
  course_id VARCHAR(20) REFERENCES iu_catalog.course_details(course_id) ON DELETE CASCADE,
  year INTEGER NOT NULL,
  slots_filled INTEGER NOT NULL,
  total_slots INTEGER NOT NULL,
  slots_filled_time INTEGER,
  avg_gpa FLOAT,
  avg_hours_spent FLOAT,
  avg_rating FLOAT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(course_id, year)
);

-- Create chat_sessions table
CREATE TABLE iu_catalog.chat_sessions (
  session_id VARCHAR(100) PRIMARY KEY,
  user_id VARCHAR(20) REFERENCES iu_catalog.login(user_id) ON DELETE CASCADE,
  title VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create chat_messages table
CREATE TABLE iu_catalog.chat_messages (
  id SERIAL PRIMARY KEY,
  session_id VARCHAR(100) REFERENCES iu_catalog.chat_sessions(session_id) ON DELETE CASCADE,
  user_id VARCHAR(20) REFERENCES iu_catalog.login(user_id),
  message_type VARCHAR(10) CHECK (message_type IN ('user', 'assistant')),
  content TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  metadata JSONB
);

-- Create career_paths table
CREATE TABLE iu_catalog.career_paths (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  skills_required JSONB,
  recommended_courses JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create course_reviews with correct references
CREATE TABLE iu_catalog.course_reviews (
  id SERIAL PRIMARY KEY,
  course_id VARCHAR(20) REFERENCES iu_catalog.course_details(course_id) ON DELETE CASCADE,
  user_id VARCHAR(20) REFERENCES iu_catalog.student_profile(user_id) ON DELETE CASCADE,
  rating INTEGER CHECK (rating BETWEEN 1 AND 5),
  review_text TEXT,
  workload_rating INTEGER CHECK (workload_rating BETWEEN 1 AND 5),
  difficulty_rating INTEGER CHECK (difficulty_rating BETWEEN 1 AND 5),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, course_id)
);

-- Create student_skills with correct references
CREATE TABLE iu_catalog.student_skills (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(20) REFERENCES iu_catalog.student_profile(user_id) ON DELETE CASCADE,
  skill_name VARCHAR(100) NOT NULL,
  proficiency_level INTEGER CHECK (proficiency_level BETWEEN 1 AND 5),
  acquired_from VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, skill_name)
);

-- Create recommendation_history with correct references
CREATE TABLE iu_catalog.recommendation_history (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(20) REFERENCES iu_catalog.student_profile(user_id) ON DELETE CASCADE,
  courses_recommended JSONB,
  selection_made JSONB,
  query_text TEXT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector search after the table exists
CREATE INDEX course_embedding_idx ON iu_catalog.course_details USING ivfflat (embedding vector_cosine_ops);



NOT EXISTS (
            SELECT 1
            FROM unnest(e.prerequisites) AS prereq
            WHERE prereq NOT IN (
                SELECT course_id 
                FROM completed_courses
            )
        ))