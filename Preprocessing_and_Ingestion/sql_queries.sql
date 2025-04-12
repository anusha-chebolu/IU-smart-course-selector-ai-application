WITH 
-- Get student's information
student_info AS (
    SELECT 
        user_id, 
        remaining_credits,
        upcoming_semester
    FROM 
        iu_catalog.student_profile
    WHERE 
        user_id = 'k11'
),


-- Get student's completed courses
completed_courses AS (
    SELECT 
        course_id
    FROM 
        iu_catalog.completed_courses
    WHERE 
        user_id = 'k11'
),



-- Get eligible courses (filtering by max_credits, not completed, offered in upcoming semester)
eligible_courses AS (
    SELECT 
        c.course_id,
        c.course_name,
        c.department,
        c.max_credits,
		c.offered_semester,
        c.prerequisites,
        c.course_title,
        c.course_description,
        c.embedding
    FROM 
        iu_catalog.course_details c
    CROSS JOIN 
        student_info s
    WHERE 
        (c.max_credits <= s.remaining_credits OR c.min_credits <= s.remaining_credits)
        AND c.offered_semester ILIKE '%' || s.upcoming_semester || '%'
        AND NOT EXISTS (
            SELECT 1 
            FROM completed_courses cc 
            WHERE cc.course_id = c.course_id
        )
),

-- Filter courses based on prerequisites being met
courses_with_prerequisites_met AS (
    SELECT 
        e.*
    FROM 
        eligible_courses e
    WHERE 
        -- Case 1: No prerequisites (empty array)
        e.prerequisites = '{}'
        OR
        -- Case 2: All prerequisites completed
		NOT EXISTS (
				-- Find any prerequisites that are NOT in completed courses
				SELECT unnest(e.prerequisites) AS prereq
				EXCEPT
				SELECT course_id FROM completed_courses
			)
)

-- Apply semantic search on filtered courses
SELECT 
    c.course_id,
    c.course_name,
    c.department,
    c.max_credits,
    c.course_title,
    c.course_description,
    (1-(c.embedding <=> 'QUERY_EMBEDDING'::VECTOR)) AS similarity
FROM 
    courses_with_prerequisites_met c
ORDER BY 
    similarity DESC
LIMIT 30;



-----------------------------------------

WITH 
-- Get student's information
student_info AS (
    SELECT 
        user_id, 
        remaining_credits,
        upcoming_semester
    FROM 
        iu_catalog.student_profile
    WHERE 
        user_id = 'k11'
),


-- Get student's completed courses
completed_courses AS (
    SELECT 
        course_id
    FROM 
        iu_catalog.completed_courses
    WHERE 
        user_id = 'k11'
),



-- Get eligible courses (filtering by max_credits, not completed, offered in upcoming semester)
eligible_courses AS (
    SELECT 
        c.course_id,
        c.course_name,
        c.department,
        c.max_credits,
		c.offered_semester,
        c.prerequisites,
        c.course_title,
        c.course_description,
        c.embedding
    FROM 
        iu_catalog.course_details c
    CROSS JOIN 
        student_info s
    WHERE 
        (c.max_credits <= s.remaining_credits OR c.min_credits <= s.remaining_credits)
        AND c.offered_semester ILIKE '%' || s.upcoming_semester || '%'
        AND NOT EXISTS (
            SELECT 1 
            FROM completed_courses cc 
            WHERE cc.course_id = c.course_id
        )
),

-- Filter courses based on prerequisites being met
courses_with_prerequisites_met AS (
    SELECT 
        e.*
    FROM 
        eligible_courses e
    WHERE 
        -- Case 1: No prerequisites (empty array)
        e.prerequisites = '{}'
        OR
        -- Case 2: All prerequisites completed
		NOT EXISTS (
				-- Find any prerequisites that are NOT in completed courses
				SELECT unnest(e.prerequisites) AS prereq
				EXCEPT
				SELECT course_id FROM completed_courses
			)
)

-- Apply semantic search on filtered courses
SELECT 
    c.course_id,
    c.course_name,
    c.department,
    c.max_credits,
    c.course_title,
    c.course_description,
    (1-(c.embedding <=> 'QUERY_EMBEDDING'::vector)) AS similarity
FROM 
    courses_with_prerequisites_met c
ORDER BY 
    similarity DESC
LIMIT 30;