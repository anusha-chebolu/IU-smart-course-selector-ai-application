# models.py
from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime, Boolean, Table, TEXT, ARRAY, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Login(Base):
    __tablename__   = 'login'
    __table_args__  = {'schema': 'iu_catalog'}
    
    user_id         = Column(String(20), primary_key=True)
    first_name      = Column(String(100), nullable=False)
    last_name       = Column(String(100), nullable=False)
    email           = Column(String(255), unique=True, nullable=False)
    password        = Column(String(255), nullable=False)
    created_at      = Column(DateTime, server_default=func.now())
    last_login      = Column(DateTime)


class StudentProfile(Base):
    __tablename__   = 'student_profile'
    __table_args__  = {'schema': 'iu_catalog'}
    
    user_id             = Column(String(20), primary_key=True)
    degree_type         = Column(String(50), nullable=False)
    major               = Column(String(100), nullable=False)
    enrollment_type     = Column(String(20), nullable=False)
    upcoming_semester   = Column(String(10), nullable=False)
    gpa                 = Column(Float(precision=3, asdecimal=True))
    total_credits       = Column(Integer, nullable=False, default=0)
    completed_credits   = Column(Integer, nullable=False, default=0)
    remaining_credits   = Column(Integer, nullable=False, default=0)
    time_availability   = Column(JSONB)
    created_at          = Column(DateTime, server_default=func.now())
    updated_at          = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint("enrollment_type IN ('Full-time', 'Part-time')"),
        CheckConstraint("upcoming_semester IN ('fall', 'spring', 'winter', 'summer')"),
        {'schema': 'iu_catalog'}
    )

class CourseDetails(Base):
    __tablename__   = 'course_details'
    __table_args__  = {'schema': 'iu_catalog'}
    
    id              = Column(Integer, primary_key=True)
    course_id       = Column(String(20), unique=True, nullable=False)
    course_name     = Column(String(255), nullable=False)
    department      = Column(String(100), nullable=False)
    min_credits     = Column(Integer, nullable=False)
    max_credits     = Column(Integer, nullable=False)
    prerequisites   = Column(ARRAY(String), default=[])
    offered_semester = Column(String(100), nullable=False)
    course_title    = Column(String(255), nullable=False)
    course_description = Column(TEXT)
    course_details  = Column(JSONB)
    embedding       = Column(Vector(384))  # Using our custom VECTOR type
    created_at      = Column(DateTime, server_default=func.now())
    updated_at      = Column(DateTime, server_default=func.now(), onupdate=func.now())
    

class CompletedCourse(Base):
    __tablename__   = 'completed_courses'
    __table_args__  = {'schema': 'iu_catalog'}
    
    id          = Column(Integer, primary_key=True)
    user_id     = Column(String(20), ForeignKey('iu_catalog.student_profile.user_id', ondelete='CASCADE'), nullable=False)
    course_id   = Column(String(20), ForeignKey('iu_catalog.course_details.course_id', ondelete='CASCADE'), nullable=False)
    semester    = Column(String(20), nullable=False)
    year        = Column(Integer, nullable=False)
    completed_date = Column(DateTime)
    gpa         = Column(Float(precision=3, asdecimal=False))
    credits     = Column(Integer, nullable=False)
    
    # Unique constraint
    __table_args__ = (
        sa.UniqueConstraint('user_id', 'course_id'),
        {'schema': 'iu_catalog'}
    )

class MyCourseList(Base):
    __tablename__   = 'my_course_list'
    __table_args__  = {'schema': 'iu_catalog'}
    
    id              = Column(Integer, primary_key=True)
    user_id         = Column(String(20), ForeignKey('iu_catalog.student_profile.user_id', ondelete='CASCADE'), nullable=False)
    course_id       = Column(String(20), ForeignKey('iu_catalog.course_details.course_id', ondelete='CASCADE'), nullable=False)
    semester        = Column(String(20))
    year            = Column(Integer)
    priority        = Column(Integer, default=0)
    notes           = Column(TEXT)
    recommendation_id = Column(Integer, ForeignKey('iu_catalog.recommendation_history.id'))
    added_date      = Column(DateTime, server_default=func.now())
    
    # Unique constraint
    __table_args__ = (
        sa.UniqueConstraint('user_id', 'course_id'),
        {'schema': 'iu_catalog'}
    )

class CourseTrends(Base):
    __tablename__ = 'course_trends'
    __table_args__ = {'schema': 'iu_catalog'}
    
    id          = Column(Integer, primary_key=True)
    course_id   = Column(String(20), ForeignKey('iu_catalog.course_details.course_id', ondelete='CASCADE'), nullable=False)
    year        = Column(Integer, nullable=False)
    slots_filled    = Column(Integer, nullable=False)
    total_slots     = Column(Integer, nullable=False)
    avg_rating      = Column(Float)
    slots_filled_time = Column(Integer)
    avg_gpa      = Column(Float)
    avg_hours_spent     = Column(Float)
    created_at      = Column(DateTime, server_default=func.now())
    
    # Unique constraint
    __table_args__ = (
        sa.UniqueConstraint('course_id', 'year'),
        {'schema': 'iu_catalog'}
    )

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    __table_args__ = {'schema': 'iu_catalog'}
    
    session_id = Column(String(100), primary_key=True)
    user_id = Column(String(20), ForeignKey('iu_catalog.login.user_id', ondelete='CASCADE'))
    title = Column(String(255))
    created_at = Column(DateTime, server_default=func.now())
    last_activity = Column(DateTime, server_default=func.now(), onupdate=func.now())
   

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    __table_args__ = {'schema': 'iu_catalog'}
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), ForeignKey('iu_catalog.chat_sessions.session_id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(20), ForeignKey('iu_catalog.login.user_id'))
    message_type = Column(String(10), nullable=False)
    content = Column(TEXT, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    meta_data = Column(JSONB)
    
   
    
    # Constraints
    __table_args__ = (
        CheckConstraint("message_type IN ('user', 'assistant')"),
        {'schema': 'iu_catalog'}
    )

class CareerPath(Base):
    __tablename__ = 'career_paths'
    __table_args__ = {'schema': 'iu_catalog'}
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(TEXT)
    skills_required = Column(JSONB)
    recommended_courses = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class CourseReview(Base):
    __tablename__ = 'course_reviews'
    __table_args__ = {'schema': 'iu_catalog'}
    
    id = Column(Integer, primary_key=True)
    course_id = Column(String(20), ForeignKey('iu_catalog.course_details.course_id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(20), ForeignKey('iu_catalog.student_profile.user_id', ondelete='CASCADE'), nullable=False)
    rating = Column(Integer)
    review_text = Column(TEXT)
    workload_rating = Column(Integer)
    difficulty_rating = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())
    
    
    
    # Constraints
    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 5"),
        CheckConstraint("workload_rating BETWEEN 1 AND 5"),
        CheckConstraint("difficulty_rating BETWEEN 1 AND 5"),
        sa.UniqueConstraint('user_id', 'course_id'),
        {'schema': 'iu_catalog'}
    )

class StudentSkill(Base):
    __tablename__ = 'student_skills'
    __table_args__ = {'schema': 'iu_catalog'}
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(20), ForeignKey('iu_catalog.student_profile.user_id', ondelete='CASCADE'), nullable=False)
    skill_name = Column(String(100), nullable=False)
    proficiency_level = Column(Integer)
    acquired_from = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
 
    
    # Constraints
    __table_args__ = (
        CheckConstraint("proficiency_level BETWEEN 1 AND 5"),
        sa.UniqueConstraint('user_id', 'skill_name'),
        {'schema': 'iu_catalog'}
    )

class RecommendationHistory(Base):
    __tablename__ = 'recommendation_history'
    __table_args__ = {'schema': 'iu_catalog'}
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(20), ForeignKey('iu_catalog.student_profile.user_id', ondelete='CASCADE'), nullable=False)
    courses_recommended = Column(JSONB)
    selection_made = Column(JSONB)
    query_text = Column(TEXT)
    timestamp = Column(DateTime, server_default=func.now())
    
   