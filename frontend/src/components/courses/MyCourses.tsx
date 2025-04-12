import React from "react";
import { useGetUserCoursesQuery } from "../../services/mycourses_api";
import CourseList from "./CourseList";

const MyCourses: React.FC = () => {
  // Fetch user's courses
  const { data: userCourses = [], isLoading, error } = useGetUserCoursesQuery();

  return (
    <div className="my-courses w-full h-full overflow-y-auto bg-white p-6">
      <h1 className="text-2xl font-bold mb-4">My Courses</h1>

      {/* Loading State */}
      {isLoading && (
        <div className="text-center py-10">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          <p className="mt-2">Loading your courses...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-4 bg-red-50 text-red-700 rounded-lg mb-6">
          <h3 className="font-bold">Error Loading Courses</h3>
          <p>
            There was a problem loading your courses. Please try again later.
          </p>
        </div>
      )}

      {/* No Courses State */}
      {!isLoading && !error && userCourses.length === 0 && (
        <div className="text-center py-10 text-gray-500">
          <p>You haven't added any courses yet.</p>
          <p className="mt-2">
            Start exploring and adding courses to your list!
          </p>
        </div>
      )}

      {/* Courses List */}
      {!isLoading && !error && userCourses.length > 0 && (
        <>
          <CourseList
            courses={userCourses.map((myCourse) => myCourse.courseDetails)}
            totalCount={userCourses.length}
          />
        </>
      )}
    </div>
  );
};

export default MyCourses;
