import json
import psycopg2
import requests
import time
import logging
from psycopg2.extras import Json
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("course_import.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Function to get embeddings from API
def get_embedding(text):
    """Call the embedding API to convert text into vector embeddings"""
    # Configuration for the embedding API
    api_url = "http://127.0.0.1:8001/embed"
    max_retries = 3
    retry_delay = 2
    
    # Ensure we have text to embed
    if not text or text.strip() == "":
        logger.warning("Empty text provided for embedding")
        return []
    
    # Prepare a rich text representation for embedding
    embedding_text = text.strip()
    
    # Try to get embeddings with retries for resilience
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url, 
                json={"texts": [embedding_text]},
                timeout=10
            )
            
            if response.status_code == 200:
                embedding_data = response.json()
                if "embeddings" in embedding_data:
                    logger.info(f"Successfully generated embedding (dimension: {len(embedding_data['embedding'])})")
                    return embedding_data["embeddings"]
                else:
                    logger.error(f"Embedding API response missing 'embedding' field: {embedding_data}")
            else:
                logger.error(f"Embedding API error: Status {response.status_code}, {response.text}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt+1}: {e}")
        
        # Wait before retrying
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    logger.error(f"Failed to get embedding after {max_retries} attempts")
    return []

# Function to connect to PostgreSQL
def get_db_connection(db_config):
    """Establish connection to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

# Function to process a single course
def process_course(course_data, conn):
    """Process a single course JSON object and insert into the database"""
    cursor = conn.cursor()
    
    try:
        # Extract basic course information
        course_id = course_data.get("courseId")
        if not course_id:
            logger.error("Missing courseId in data, skipping record")
            return False
            
        # Extract other fields
        catalog_number = course_data.get("catalogNumber", "")
        subject = course_data.get("subject", "")
        course_name = f"{subject} {catalog_number}".strip()
        
        # Extract department from subject (before the hyphen if present)
        department = subject.split("-")[0] if "-" in subject else subject
        
        # Get credits
        mincredits = course_data.get("minCredits", 0)
        maxcredits = course_data.get("maxCredits", 0)
        # Get prerequisites
        prerequisites = course_data.get("prerequisites", [])
        
        # Get course details
        course_details_info = course_data.get("courseDetails", {})
        offered_semester = course_details_info.get("typicallyOffered", "")
        title = course_data.get("title", "")
        description = course_details_info.get("description", "")
        
        # Create comprehensive course details JSON
        course_details_json = {
            "attributes": course_details_info.get("attributes", []),
            "academicCareer": course_data.get("car", ""),
            "effectiveDate": course_data.get("effdt", ""),
            "courseOfferNumber": course_data.get("courseOfferNumber", ""),
            "courseTopicId": course_data.get("courseTopicId", ""),
            "institution": course_data.get("inst", ""),
            "classes": []
        }
        
        # Add class information
        additional_info = course_data.get("courseAdditionalInfo", {})
        classes = additional_info.get("classes", [])
        
        for class_info in classes:
            class_details = {
                "classNumber": class_info.get("classNbr", ""),
                "component": class_info.get("componentDescription", ""),
                "sessionDescription": class_info.get("sessionDescr", ""),
                "sessionStartDate": class_info.get("sessionStartDateString", ""),
                "sessionEndDate": class_info.get("sessionEndDateString", ""),
                "location": class_info.get("locationDescription", ""),
                "modeOfInstruction": class_info.get("modeOfInstruction", ""),
                "totalSeats": class_info.get("totalSeats", 0),
                "openSeats": class_info.get("openSeats", 0),
                "waitlistTotal": class_info.get("waitlistTotal", 0),
                "instructors": [],
                "meetings": [],
                "notes": class_info.get("classNotes", []),
                "enrollmentRequirements": class_info.get("enrollmentRequirements", []),
                "departmentConsent": class_info.get("departmentConsentRequired", False),
                "instructorConsent": class_info.get("instructorConsentRequired", False)
            }
            
            # Add instructor information
            for instructor in class_info.get("primaryInstructors", []):
                class_details["instructors"].append({
                    "name": instructor.get("fullName", ""),
                    "firstName": instructor.get("firstName", ""),
                    "lastName": instructor.get("lastName", ""),
                    "role": instructor.get("role", "")
                })
            
            # Add meeting information
            for meeting in class_info.get("meetings", []):
                meeting_info = {
                    "beginDate": meeting.get("beginDateString", ""),
                    "endDate": meeting.get("endDateString", ""),
                    "building": meeting.get("buildingName", ""),
                    "room": meeting.get("room", ""),
                    "startTime": meeting.get("startTimeString", ""),
                    "endTime": meeting.get("endTimeString", ""),
                    "days": []
                }
                
                # Extract meeting days
                for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                    if meeting.get(day):
                        meeting_info["days"].append(day.capitalize())
                
                class_details["meetings"].append(meeting_info)
            
            course_details_json["classes"].append(class_details)
        
        # Generate embedding for the course - combine relevant text fields
        embedding_text = f"Title: {title}. Description: {description}."
        embedding = get_embedding(embedding_text)
        
        # Insert into the course_details table
        query = """
        INSERT INTO iu_catalog.course_details 
        (course_id, course_name, department, credits, prerequisites, offered_semester, 
         course_title, course_description, course_details, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (course_id) DO UPDATE SET
        course_name = EXCLUDED.course_name,
        department = EXCLUDED.department,
        credits = EXCLUDED.credits,
        prerequisites = EXCLUDED.prerequisites,
        offered_semester = EXCLUDED.offered_semester,
        course_title = EXCLUDED.course_title,
        course_description = EXCLUDED.course_description,
        course_details = EXCLUDED.course_details,
        embedding = EXCLUDED.embedding,
        updated_at = CURRENT_TIMESTAMP
        """
        
        cursor.execute(query, (
            course_id, 
            course_name, 
            department, 
            credits, 
            prerequisites, 
            offered_semester, 
            title, 
            description, 
            Json(course_details_json),
            embedding
        ))
        conn.commit()
        logger.info(f"Successfully inserted/updated course: {course_id} - {course_name}")
        return True
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error processing course {course_data.get('courseId', 'Unknown')}: {str(e)}")
        return False
    finally:
        cursor.close()

# Function to load multiple courses from a JSON file
def load_courses_from_file(file_path, db_config):
    """Load courses from a JSON file into the database"""
    try:
        # Load JSON data
        logger.info(f"Loading course data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Connect to PostgreSQL
        conn = get_db_connection(db_config)
        
        success_count = 0
        error_count = 0
        
        data = data.get("courses", [])

        # Process each course
        courses = data if isinstance(data, list) else [data]
        total_courses = len(courses)
        logger.info(f"Found {total_courses} courses to process")
        
        for i, course in enumerate(courses):
            logger.info(f"Processing course {i+1}/{total_courses}")
            if process_course(course, conn):
                success_count += 1
            else:
                error_count += 1
            
            # Occasional progress updates for large imports
            if (i+1) % 100 == 0 or i+1 == total_courses:
                logger.info(f"Progress: {i+1}/{total_courses} courses processed. Success: {success_count}, Failed: {error_count}")
        
        conn.close()
        logger.info(f"Course data loading completed. Successful: {success_count}, Failed: {error_count}")
        
    except Exception as e:
        logger.error(f"Error in load_courses_from_file: {str(e)}")

# Command line execution
if __name__ == "__main__":

    SCRIPT_DIR  = os.path.dirname(os.path.realpath(__file__))
    SOURCE_DIR  = os.path.join(SCRIPT_DIR, "source_data")
    SOURCE_FILE_PATH = os.path.join(SOURCE_DIR, "distinct_courses.json") 
    # OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "faiss_index")

    config = {
        "file_path": SOURCE_FILE_PATH,
        "db_config": {
            "host": "localhost",
            "port": 5432,
            "database": "smart_search_course_recommendation",
            "user": "postgres",
            "password": "mz7zdz123"
        }
    }

    
    # Load courses
    load_courses_from_file(SOURCE_FILE_PATH, config['db_config'])