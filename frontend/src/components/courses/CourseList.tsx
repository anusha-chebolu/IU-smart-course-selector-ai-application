// components/CourseList.tsx
import React, { useState } from "react";
import CourseItem from "./CourseItem";
import { Course } from "../../types";

interface CourseListProps {
  courses: Course[];
  totalCount: number;
}

const CourseList: React.FC<CourseListProps> = ({ courses, totalCount }) => {
  const [visibleCount, setVisibleCount] = useState(5); // Start with 5 courses
  const [expandedCourseId, setExpandedCourseId] = useState<string | null>(null);

  const displayedCourses = courses.slice(0, visibleCount);

  const loadMoreCourses = () => {
    setVisibleCount((prev) => prev + 5); // Load 5 more each time
  };

  const toggleCourseExpansion = (courseId: string) => {
    setExpandedCourseId((prev) => (prev === courseId ? null : courseId));
  };

  return (
    <div className="course-list-container">
      <div className="flex justify-between items-center mb-6">
        <div className="font-medium">{totalCount} results</div>

        <div className="flex space-x-2">
          <button className="p-2 bg-gray-200 rounded">
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
          <button className="p-2 rounded">
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"
              />
            </svg>
          </button>
        </div>
      </div>

      {displayedCourses.length === 0 ? (
        <div className="text-center py-10 border rounded-md bg-gray-50">
          <p>No courses match your current filters.</p>
        </div>
      ) : (
        <div>
          <div className="space-y-4">
            {" "}
            {/* This adds proper spacing between each item */}
            {displayedCourses.map((course) => (
              <CourseItem
                key={course.course_id}
                course={course}
                isExpanded={expandedCourseId === course.course_id}
                onToggleExpand={() => toggleCourseExpansion(course.course_id)}
              />
            ))}
          </div>

          {/* Always visible load more button if there are more courses */}
          {visibleCount < totalCount && (
            <div className="text-center py-8 mt-6">
              <button
                onClick={loadMoreCourses}
                className="px-6 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors"
              >
                Load More
              </button>
              <p className="text-sm text-gray-500 mt-2">
                Showing {displayedCourses.length} of {totalCount} courses
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CourseList;
