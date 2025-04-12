import json
import requests
import time
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.postgresql_model import CourseTrends

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


# Main bulk loading function with batch processing
def load_trends_from_file(file_path, batch_size=25000):
    session = Session()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_trends = len(data)
        logger.info(f"Found {total_trends} courses to process")
        
        for batch_start in range(0, total_trends, batch_size):
            batch_courses = data[batch_start:batch_start+batch_size]
            trend_records = []
            for trend_data in batch_courses:
                course_id = trend_data.get("course_id", "")
                year = trend_data.get("year", "")
                total_slots = trend_data.get("total_slots", "")
                filled_slots = trend_data.get("filled_slots", "")
                time_to_fill =  trend_data.get("time_to_fill", "")
                average_gpa =  trend_data.get("average_gpa", "")
                average_hours_spent =  trend_data.get("average_hours_spent", "")
                avg_rating =  trend_data.get("rating", "")


                record = CourseTrends(
                    course_id=course_id,
                    year=year,
                    slots_filled=filled_slots,
                    total_slots=total_slots,
                    avg_rating=avg_rating,
                    slots_filled_time=time_to_fill,
                    avg_gpa      = average_gpa,
                    avg_hours_spent=average_hours_spent
                )

                trend_records.append(record)

            print(len(trend_records))
            print("Data prep done")
            session.bulk_save_objects(trend_records)
            session.commit()
            logger.info(f"Processed batch {batch_start // batch_size + 1}")

        logger.info(f"Successfully loaded {len(trend_records)} courses.")

    except Exception as e:
        session.rollback()
        logger.error(f"Error during bulk insert: {e}")
    finally:
        session.close()


# Execute the script
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    SOURCE_FILE_PATH = os.path.join(SCRIPT_DIR, "final_course_data_by_year.json")

    load_trends_from_file(SOURCE_FILE_PATH)
