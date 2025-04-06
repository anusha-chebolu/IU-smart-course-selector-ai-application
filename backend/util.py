from postgresql_model import Login, StudentProfile, CourseDetails, CompletedCourse, MyCourseList, CourseTrends, ChatSession, ChatMessage, CareerPath
from postgresql_model import CourseReview, RecommendationHistory
import requests
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, func, select, text, desc, or_, and_, exists, column

from sqlalchemy import select, func, and_, desc, cast, text
from sqlalchemy.sql.expression import or_
from sqlalchemy.sql.expression import literal, except_
import sqlalchemy as sa
import os
from sqlalchemy.orm import sessionmaker, Session


import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, func, select, text, desc, or_, and_, exists, column
from sqlalchemy.sql.expression import literal, except_
import sqlalchemy as sa
from contextlib import asynccontextmanager
from typing import Optional
from postgresql_model import Login, StudentProfile, CourseDetails, CompletedCourse, MyCourseList, CourseTrends, ChatSession, ChatMessage, CareerPath
from postgresql_model import CourseReview, RecommendationHistory
from postgresql_model import Base
import requests
from sqlalchemy import select, func, and_, desc, cast, text
from sqlalchemy.sql.expression import or_
from sqlalchemy.dialects.postgresql import ARRAY
from typing import List, Dict, Any, Optional
import requests
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "smart_search_course_recommendation",
    "user": "postgres",
    "password": "mz7zdz123"
}


# SQLAlchemy setup
DATABASE_URL = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def get_course_recommendations(user_id, query_text, session, top_n=10):
    """
    Get course recommendations for a student based on their profile, completed courses,
    and semantic similarity to the query embedding.
    
    Args:
        user_id (str): Student user ID
        query_embedding (list): Vector embedding of the query (list of 384 floating point numbers)
        top_n (int): Number of results to return (default: 30)
        db_config (dict): Database connection parameters (optional)
    
    Returns:
        list: Top N recommended courses ordered by embedding similarity
    """
    print(f"Generating course recommendations for user_id: {user_id} with query: {query_text}")
    embedding_api_url = os.environ.get("EMBEDDING_API_URL", "")
    query = ' '.join(query_text) if isinstance(query_text, list) else query_text
    try:
        session = SessionLocal()
        try:
            response = requests.post(
                embedding_api_url,
                json={"texts": [query]},
                timeout=10
            )
            response.raise_for_status()
            embedding_result = response.json()
            query_embedding = embedding_result.get("embeddings", [])[0]
            
            if not query_embedding:
                raise ValueError("Failed to generate embedding for query text")
                
            print(f"Generated embedding for query text: {len(query_embedding)} dimensions")
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating embedding: {str(e)}"
            )


        # Step 1: Get student's information
        student_info = session.query(
            StudentProfile.user_id,
            StudentProfile.remaining_credits,
            StudentProfile.upcoming_semester
        ).filter(
            StudentProfile.user_id == user_id
        ).subquery()
        
        # Step 2: Get student's completed courses
        completed_courses = session.query(
            CompletedCourse.course_id
        ).filter(
            CompletedCourse.user_id == user_id
        ).subquery()
        
        # Step 3: Find eligible courses
        eligible_courses = session.query(
            CourseDetails.course_id,
            CourseDetails.course_name,
            CourseDetails.department,
            CourseDetails.min_credits,
            CourseDetails.max_credits,
            CourseDetails.offered_semester,
            CourseDetails.prerequisites,
            CourseDetails.course_title,
            CourseDetails.course_description,
            CourseDetails.embedding
        ).join(
            student_info,
            sa.literal(True)
        ).filter(
            # Credits check
            or_(
                CourseDetails.max_credits <= student_info.c.remaining_credits,
                CourseDetails.min_credits <= student_info.c.remaining_credits
            ),
            # Semester check
            CourseDetails.offered_semester.ilike('%' + student_info.c.upcoming_semester + '%'),
            # Not already completed
            ~exists().where(
                completed_courses.c.course_id == CourseDetails.course_id
            )
        ).subquery()

        # Step 4: Filter courses based on prerequisites being met
        # Using SQLAlchemy's approach for array operation
        
        # First, create a subquery for prerequisites that must be met
        prereq_subquery = session.query(
            eligible_courses.c.course_id,
            func.unnest(eligible_courses.c.prerequisites).label('prereq')
        ).subquery()

        # Then check if all prerequisites are in completed courses
        missing_prereqs = session.query(
            prereq_subquery.c.course_id
        ).filter(
            ~exists().where(
                and_(
                    completed_courses.c.course_id == prereq_subquery.c.prereq,
                    prereq_subquery.c.prereq != None,
                    prereq_subquery.c.prereq != ''
                )
            )
        ).subquery()

        courses_with_prerequisites_met = session.query(
            eligible_courses.c.course_id,
            eligible_courses.c.course_name,
            eligible_courses.c.department,
            eligible_courses.c.min_credits,
            eligible_courses.c.max_credits,
            eligible_courses.c.course_title,
            eligible_courses.c.course_description,
            eligible_courses.c.embedding
        ).filter(
            or_(
                # Case 1: No prerequisites (empty array)
                eligible_courses.c.prerequisites == '{}',
                # Case 2: Course ID not in missing prerequisites list
                ~exists().where(
                    missing_prereqs.c.course_id == eligible_courses.c.course_id
                )
            )
        ).subquery()
        
        
        # Step 5: Apply semantic search on filtered courses
        similarity_expr = func.coalesce(
            literal(1) - (courses_with_prerequisites_met.c.embedding.cosine_distance(query_embedding)),
            0
        ).label("similarity")
        
        results = session.query(
            courses_with_prerequisites_met.c.course_id,
            courses_with_prerequisites_met.c.course_name,
            courses_with_prerequisites_met.c.department,
            courses_with_prerequisites_met.c.min_credits,
            courses_with_prerequisites_met.c.max_credits,
            courses_with_prerequisites_met.c.course_title,
            courses_with_prerequisites_met.c.course_description,
            similarity_expr
        ).order_by(
            desc("similarity")
        ).limit(top_n).all()
        
        # Convert results to dictionary format for easier handling
        courses = []
        for r in results:
            courses.append({
                'course_id': r.course_id,
                'course_name': r.course_name,
                'department': r.department,
                'max_credits': r.min_credits,
                'max_credits': r.max_credits,
                'course_title': r.course_title,
                'course_description': r.course_description,
                'similarity': float(r.similarity)
            })
            
        return courses
        
    except Exception as e:
        print(f"Error querying database: {e}")
        return []
        
    finally:
        session.close()