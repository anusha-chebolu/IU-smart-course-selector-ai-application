INSERT INTO iu_catalog.student_profile (
  user_id, degree_type, major, enrollment_type, gpa, total_credits, completed_credits, remaining_credits, time_availability, upcoming_semester
) VALUES 
('hbishai', 'Bachelor', 'Computer Science', 'Full-time', 3.75, 120, 90, 30, '{"Monday": "9am-5pm", "Tuesday": "10am-4pm", "Wednesday": "9am-1pm", "Thursday": "11am-5pm", "Friday": "1pm-6pm"}', 'fall'),

('k11', 'Master', 'Data Science', 'Part-time', 3.40, 30, 15, 15, '{"Monday": "9am-1pm", "Tuesday": "10am-5pm", "Wednesday": "1pm-5pm", "Thursday": "9am-12pm"}', 'fall'),

('vpothapr', 'PhD', 'Artificial Intelligence', 'Full-time', 4.00, 60, 18, 42, '{"Monday": "9am-6pm", "Tuesday": "10am-2pm", "Wednesday": "2pm-6pm", "Thursday": "9am-3pm", "Friday": "10am-4pm"}', 'spring'),

('srcheb', 'Bachelor', 'Mechanical Engineering', 'Part-time', 3.20, 120, 60, 60, '{"Monday": "1pm-6pm", "Tuesday": "9am-1pm", "Wednesday": "2pm-6pm", "Thursday": "9am-4pm"}', 'spring'),

('jbudigin', 'Master', 'Cybersecurity', 'Full-time', 3.85, 30, 27, 3, '{"Monday": "9am-5pm", "Tuesday": "1pm-6pm", "Wednesday": "10am-4pm", "Thursday": "9am-1pm", "Friday": "2pm-5pm"}', 'summer'),

('u006', 'Bachelor', 'Business Administration', 'Full-time', 3.10, 120, 100, 20, '{"Monday": "8am-12pm", "Tuesday": "10am-6pm", "Wednesday": "9am-1pm", "Thursday": "1pm-5pm", "Friday": "2pm-6pm"}', 'fall'),

('u007', 'Master', 'Information Systems', 'Part-time', 3.55, 30, 18, 12, '{"Monday": "12pm-4pm", "Tuesday": "9am-12pm", "Wednesday": "10am-4pm", "Thursday": "1pm-5pm", "Friday": "10am-2pm"}', 'winter'),

('u008', 'PhD', 'Biotechnology', 'Full-time', 3.95, 60, 45, 15, '{"Monday": "9am-2pm", "Tuesday": "2pm-6pm", "Wednesday": "9am-5pm", "Thursday": "11am-4pm", "Friday": "9am-12pm"}', 'spring'),

('u009', 'Bachelor', 'Electrical Engineering', 'Full-time', 3.00, 120, 80, 40, '{"Monday": "9am-3pm", "Tuesday": "10am-4pm", "Wednesday": "12pm-6pm", "Thursday": "10am-2pm", "Friday": "1pm-5pm"}', 'summer'),

('u010', 'Master', 'Finance', 'Part-time', 3.65, 30, 24, 6, '{"Monday": "10am-4pm", "Tuesday": "9am-12pm", "Wednesday": "1pm-6pm", "Thursday": "10am-3pm", "Friday": "9am-2pm"}', 'fall');



INSERT INTO iu_catalog.completed_courses (user_id, course_id, credits, year, semester, gpa, completed_date)
VALUES 
('hbishai', '098542', 3, 2024, 'Summer', 2.0, '2024-08-16'),
('hbishai', '011234', 4, 2020, 'Fall', 3.0, '2020-12-20'),
('hbishai', '011293', 3, 2023, 'Spring', 4.0, '2023-05-14'),
('hbishai', '089122', 3, 2021, 'Summer', 3.7, '2021-08-10'),
('hbishai', '092530', 3, 2021, 'Spring', 3.3, '2021-05-19'),
('hbishai', '011297', 3, 2022, 'Summer', 2.7, '2022-08-12'),
('hbishai', '097923', 3, 2023, 'Fall', 4.0, '2023-12-20'),
('hbishai', '011248', 3, 2022, 'Fall', 3.3, '2022-12-22'),
('hbishai', '097614', 4, 2020, 'Fall', 2.0, '2020-12-11'),
('hbishai', '011286', 3, 2024, 'Summer', 2.7, '2024-08-15'),
('hbishai', '011297', 3, 2022, 'Spring', 3.7, '2022-05-25'),
('hbishai', '011140', 4, 2020, 'Summer', 2.0, '2020-08-11'),
('hbishai', '011268', 3, 2024, 'Summer', 3.0, '2024-08-18'),
('hbishai', '011260', 3, 2020, 'Fall', 3.0, '2020-12-23'),
('hbishai', '011179', 3, 2022, 'Spring', 4.0, '2022-05-19'),
('hbishai', '097927', 4, 2024, 'Summer', 4.0, '2024-08-12'),
('hbishai', '098123', 3, 2022, 'Fall', 4.0, '2022-12-19'),
('hbishai', '011262', 3, 2020, 'Summer', 3.0, '2020-08-21'),
('hbishai', '097198', 4, 2020, 'Spring', 2.3, '2020-05-13'),
('hbishai', '011266', 3, 2022, 'Fall', 1.0, '2022-12-12'),
('hbishai', '011235', 4, 2024, 'Spring', 3.7, '2024-05-20'),
('hbishai', '011139', 4, 2024, 'Fall', 4.0, '2024-12-20'),
('hbishai', '097634', 3, 2024, 'Summer', 1.0, '2024-08-22'),
('hbishai', '080675', 3, 2022, 'Spring', 3.3, '2022-05-25'),
('hbishai', '093393', 4, 2023, 'Summer', 4.0, '2023-08-10'),
('hbishai', '080676', 3, 2024, 'Fall', 3.0, '2024-12-16'),
('hbishai', '098051', 4, 2024, 'Summer', 2.3, '2024-08-24'),
('k11', '096035', 3, 2021, 'Spring', 2.7, '2021-05-14'),
('k11', '011291', 3, 2020, 'Summer', 3.0, '2020-08-15'),
('k11', '011297', 3, 2021, 'Summer', 3.7, '2021-08-10'),
('k11', '088550', 3, 2022, 'Summer', 3.3, '2022-08-11'),
('k11', '089122', 3, 2020, 'Spring', 3.3, '2020-05-18'),
('vpothapr', '080675', 3, 2020, 'Fall', 2.3, '2020-12-24'),
('vpothapr', '098123', 3, 2023, 'Summer', 3.0, '2023-08-18'),
('vpothapr', '011293', 3, 2023, 'Spring', 2.3, '2023-05-11'),
('vpothapr', '097634', 3, 2022, 'Fall', 2.0, '2022-12-19'),
('vpothapr', '089122', 3, 2023, 'Spring', 3.0, '2023-05-18'),
('vpothapr', '098036', 3, 2023, 'Fall', 2.7, '2023-12-21'),
('srcheb', '011297', 3, 2020, 'Fall', 3.3, '2020-12-10'),
('srcheb', '098036', 3, 2020, 'Spring', 3.7, '2020-05-14'),
('srcheb', '011266', 3, 2020, 'Summer', 2.3, '2020-08-17'),
('srcheb', '097927', 4, 2022, 'Summer', 4.0, '2022-08-24'),
('srcheb', '011286', 3, 2021, 'Summer', 2.3, '2021-08-13'),
('srcheb', '011140', 4, 2022, 'Summer', 3.7, '2022-08-21'),
('srcheb', '011234', 4, 2022, 'Fall', 3.0, '2022-12-10'),
('srcheb', '011297', 3, 2021, 'Spring', 3.7, '2021-05-24'),
('jbudigin', '011266', 3, 2021, 'Fall', 2.7, '2021-12-23'),
('jbudigin', '097634', 3, 2022, 'Summer', 2.3, '2022-08-21'),
('jbudigin', '011234', 4, 2024, 'Fall', 2.0, '2024-12-14'),
('jbudigin', '011268', 3, 2022, 'Spring', 2.0, '2022-05-16'),
('jbudigin', '011140', 4, 2022, 'Summer', 4.0, '2022-08-19'),
('jbudigin', '093393', 4, 2022, 'Fall', 2.7, '2022-12-24'),
('jbudigin', '011297', 3, 2022, 'Spring', 2.0, '2022-05-16'),
('jbudigin', '080675', 3, 2023, 'Fall', 4.0, '2023-12-16');



INSERT INTO iu_catalog.completed_courses (user_id, course_id, credits, year, semester, gpa, completed_date)
VALUES 
('k11', '096035', 3, 2021, 'Spring', 2.7, '2021-05-14'),
('k11', '011291', 3, 2020, 'Summer', 3.0, '2020-08-15'),
('k11', '011297', 3, 2021, 'Summer', 3.7, '2021-08-10'),
('k11', '088550', 3, 2022, 'Summer', 3.3, '2022-08-11'),
('k11', '089122', 3, 2020, 'Spring', 3.3, '2020-05-18'),
('vpothapr', '080675', 3, 2020, 'Fall', 2.3, '2020-12-24'),
('vpothapr', '098123', 3, 2023, 'Summer', 3.0, '2023-08-18'),
('vpothapr', '011293', 3, 2023, 'Spring', 2.3, '2023-05-11'),
('vpothapr', '097634', 3, 2022, 'Fall', 2.0, '2022-12-19'),
('vpothapr', '089122', 3, 2023, 'Spring', 3.0, '2023-05-18'),
('vpothapr', '098036', 3, 2023, 'Fall', 2.7, '2023-12-21'),
('srcheb', '011297', 3, 2020, 'Fall', 3.3, '2020-12-10'),
('srcheb', '098036', 3, 2020, 'Spring', 3.7, '2020-05-14'),
('srcheb', '011266', 3, 2020, 'Summer', 2.3, '2020-08-17'),
('srcheb', '097927', 4, 2022, 'Summer', 4.0, '2022-08-24'),
('srcheb', '011286', 3, 2021, 'Summer', 2.3, '2021-08-13'),
('srcheb', '011140', 4, 2022, 'Summer', 3.7, '2022-08-21'),
('srcheb', '011234', 4, 2022, 'Fall', 3.0, '2022-12-10'),
('srcheb', '011293', 3, 2021, 'Spring', 3.7, '2021-05-24'),
('jbudigin', '011266', 3, 2021, 'Fall', 2.7, '2021-12-23'),
('jbudigin', '097634', 3, 2022, 'Summer', 2.3, '2022-08-21'),
('jbudigin', '011234', 4, 2024, 'Fall', 2.0, '2024-12-14'),
('jbudigin', '011268', 3, 2022, 'Spring', 2.0, '2022-05-16'),
('jbudigin', '011140', 4, 2022, 'Summer', 4.0, '2022-08-19'),
('jbudigin', '093393', 4, 2022, 'Fall', 2.7, '2022-12-24'),
('jbudigin', '011297', 3, 2022, 'Spring', 2.0, '2022-05-16'),
('jbudigin', '080675', 3, 2023, 'Fall', 4.0, '2023-12-16');