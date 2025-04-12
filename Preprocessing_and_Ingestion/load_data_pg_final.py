import json
import requests
import time
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.postgresql_model import CourseDetails

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("course_import.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# DB Configuration
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
Session = sessionmaker(bind=engine)

# Function to get embeddings in batch from API
def get_batch_embeddings(texts):
    api_url = "http://127.0.0.1:8001/embed"
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json={"texts": texts}, timeout=20)
            if response.status_code == 200:
                embedding_data = response.json()
                if "embeddings" in embedding_data:
                    return embedding_data["embeddings"]
                else:
                    logger.error(f"Embedding API missing 'embeddings': {embedding_data}")
            else:
                logger.error(f"Embedding API error: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt+1}: {e}")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    return [[] for _ in texts]

# Main bulk loading function with batch processing
def load_courses_from_file(file_path, batch_size=100):
    session = Session()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        courses = data.get("courses", [])
        total_courses = len(courses)
        logger.info(f"Found {total_courses} courses to process")

        for batch_start in range(0, total_courses, batch_size):
            batch_courses = courses[batch_start:batch_start+batch_size]
            embedding_texts = [
                f"Title: {course.get('title', '')}. Description: {course.get('courseDetails', {}).get('description', '')}."
                for course in batch_courses
            ]

            embeddings = get_batch_embeddings(embedding_texts)
            course_records = []

            for course_data, embedding in zip(batch_courses, embeddings):
                course_id = course_data.get("courseId")
                if not course_id:
                    logger.warning("Skipping course with missing courseId")
                    continue

                subject = course_data.get("subject", "")
                catalog_number = course_data.get("catalogNumber", "")
                course_name = f"{subject} {catalog_number}".strip()
                department = subject.split("-")[0] if "-" in subject else subject
                mincredits = course_data.get("minCredits", 0)
                maxcredits = course_data.get("maxCredits", 0)
                prerequisites = course_data.get("prerequisites", [])

                course_details_info = course_data.get("courseDetails", {})
                offered_semester = course_details_info.get("typicallyOffered", "")
                title = course_data.get("title", "")
                description = course_details_info.get("description", "")

                course_details_json = {
                    "attributes": course_details_info.get("attributes", []),
                    "academicCareer": course_data.get("car", ""),
                    "effectiveDate": course_data.get("effdt", ""),
                    "courseOfferNumber": course_data.get("courseOfferNumber", ""),
                    "courseTopicId": course_data.get("courseTopicId", ""),
                    "institution": course_data.get("inst", ""),
                    "classes": course_data.get("courseAdditionalInfo", {}).get("classes", [])
                }

                record = CourseDetails(
                    course_id=course_id,
                    course_name=course_name,
                    department=department,
                    min_credits=mincredits,
                    max_credits=maxcredits,
                    prerequisites=prerequisites,
                    offered_semester=offered_semester,
                    course_title=title,
                    course_description=description,
                    course_details=course_details_json,
                    embedding=embedding
                )

                course_records.append(record)

            print("Data prep done")
            session.bulk_save_objects(course_records)
            session.commit()
            logger.info(f"Processed batch {batch_start // batch_size + 1} / {((total_courses - 1) // batch_size) + 1}")

        logger.info(f"Successfully loaded {total_courses} courses.")

    except Exception as e:
        session.rollback()
        logger.error(f"Error during bulk insert: {e}")
    finally:
        session.close()


# Execute the script
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    SOURCE_DIR = os.path.join(SCRIPT_DIR, "source_data")
    SOURCE_FILE_PATH = os.path.join(SOURCE_DIR, "distinct_courses.json")

    load_courses_from_file(SOURCE_FILE_PATH)
