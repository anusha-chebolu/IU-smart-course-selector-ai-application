// // components/CourseSearch.tsx
// import React, { useState, useEffect } from "react";
// import { useGetFilteredCoursesQuery } from "../../services/courses";
// import CourseList from "./CourseList";

// const CourseSearch: React.FC = () => {
//   const [filters, setFilters] = useState({
//     campus: "",
//     term: "",
//     department: "",
//     courseName: "",
//     maxCredits: undefined as number | undefined,
//     semesters: [] as string[],
//   });

//   // Get filtered courses
//   const { data: courses = [], isLoading } = useGetFilteredCoursesQuery(filters);

//   // Get filter options
//   const [filterOptions, setFilterOptions] = useState({
//     campuses: [] as string[],
//     terms: [] as string[],
//     departments: [] as string[],
//     semesters: ["Spring", "Summer", "Fall"] as string[],
//     credits: [1, 2, 3, 4, 6] as number[],
//   });

//   // Extract filter options from courses
//   useEffect(() => {
//     if (courses.length > 0) {
//       const campusSet = new Set<string>();
//       const termSet = new Set<string>();
//       const departmentSet = new Set<string>();
//       const creditSet = new Set<number>();

//       courses.forEach((course) => {
//         // Department
//         if (course.department) {
//           departmentSet.add(course.department);
//         }

//         // Credits
//         if (course.max_credits) {
//           creditSet.add(course.max_credits);
//         }

//         // Campus and Term
//         course.course_details?.classes.forEach((cls) => {
//           if (cls.campus) campusSet.add(cls.campus);
//           if (cls.strm) termSet.add(cls.strm);
//         });
//       });

//       setFilterOptions((prev) => ({
//         ...prev,
//         campuses: Array.from(campusSet).sort(),
//         terms: Array.from(termSet).sort(),
//         departments: Array.from(departmentSet).sort(),
//         credits: Array.from(creditSet).sort(),
//       }));
//     }
//   }, [courses]);

//   const handleFilterChange = (name: string, value: any) => {
//     setFilters((prev) => ({ ...prev, [name]: value }));
//   };

//   const handleCreditFilterChange = (
//     e: React.ChangeEvent<HTMLSelectElement>
//   ) => {
//     const value = e.target.value ? parseInt(e.target.value) : undefined;
//     setFilters((prev) => ({ ...prev, maxCredits: value }));
//   };

//   const handleSemesterToggle = (semester: string) => {
//     setFilters((prev) => {
//       const currentSemesters = [...(prev.semesters || [])];
//       if (currentSemesters.includes(semester)) {
//         return {
//           ...prev,
//           semesters: currentSemesters.filter((s) => s !== semester),
//         };
//       } else {
//         return {
//           ...prev,
//           semesters: [...currentSemesters, semester],
//         };
//       }
//     });
//   };

//   return (
//     <div className="course-search w-full h-full overflow-y-auto bg-white p-6">
//       <h1 className="text-2xl font-bold mb-4">Course Search</h1>

//       <p className="mb-4">
//         Select <strong>Campus</strong> then <strong>Term</strong> to continue
//       </p>

//       <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
//         <div>
//           <label className="block font-medium text-gray-700 mb-1">Campus</label>
//           <select
//             className="w-full p-2 border border-gray-300 rounded-md"
//             value={filters.campus}
//             onChange={(e) => handleFilterChange("campus", e.target.value)}
//           >
//             <option value="">Select Campus</option>
//             {filterOptions.campuses.map((campus) => (
//               <option key={campus} value={campus}>
//                 {campus === "BL" ? "IU Bloomington" : campus}
//               </option>
//             ))}
//           </select>
//         </div>

//         <div>
//           <label className="block font-medium text-gray-700 mb-1">Term</label>
//           <select
//             className="w-full p-2 border border-gray-300 rounded-md"
//             value={filters.term}
//             onChange={(e) => handleFilterChange("term", e.target.value)}
//           >
//             <option value="">Select Term</option>
//             {filterOptions.terms.map((term) => (
//               <option key={term} value={term}>
//                 {term === "4249" ? "Fall 2024" : term}
//               </option>
//             ))}
//           </select>
//         </div>

//         <div>
//           <label className="block font-medium text-gray-700 mb-1">
//             Department/Subject
//           </label>
//           <select
//             className="w-full p-2 border border-gray-300 rounded-md"
//             value={filters.department}
//             onChange={(e) => handleFilterChange("department", e.target.value)}
//           >
//             <option value="">All</option>
//             {filterOptions.departments.map((dept) => (
//               <option key={dept} value={dept}>
//                 {dept}
//               </option>
//             ))}
//           </select>
//         </div>
//       </div>

//       <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
//         <div>
//           <label className="block font-medium text-gray-700 mb-1">
//             Search by Course Name
//           </label>
//           <div className="flex">
//             <input
//               type="text"
//               className="flex-grow p-2 border border-gray-300 rounded-l-md"
//               placeholder="E.g., BUEX-V 594, Digital Transformation"
//               value={filters.courseName || ""}
//               onChange={(e) => handleFilterChange("courseName", e.target.value)}
//             />
//             <button
//               className="bg-blue-600 text-white px-4 py-2 rounded-r-md"
//               onClick={() => {
//                 /* Search handled on input change */
//               }}
//             >
//               Search
//             </button>
//           </div>
//         </div>

//         <div>
//           <label className="block font-medium text-gray-700 mb-1">
//             Max Credits
//           </label>
//           <select
//             className="w-full p-2 border border-gray-300 rounded-md"
//             value={filters.maxCredits?.toString() || ""}
//             onChange={handleCreditFilterChange}
//           >
//             <option value="">Any</option>
//             {filterOptions.credits.map((credit) => (
//               <option key={credit} value={credit.toString()}>
//                 {credit} Credit{credit !== 1 ? "s" : ""}
//               </option>
//             ))}
//           </select>
//         </div>
//       </div>

//       <div className="mb-6">
//         <label className="block font-medium text-gray-700 mb-1">
//           Semester Offered
//         </label>
//         <div className="flex gap-4">
//           {filterOptions.semesters.map((semester) => (
//             <div key={semester} className="flex items-center">
//               <input
//                 type="checkbox"
//                 id={`semester-${semester}`}
//                 checked={(filters.semesters || []).includes(semester)}
//                 onChange={() => handleSemesterToggle(semester)}
//                 className="mr-2"
//               />
//               <label htmlFor={`semester-${semester}`}>{semester}</label>
//             </div>
//           ))}
//         </div>
//       </div>

//       {isLoading ? (
//         <div className="text-center py-10">
//           <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
//           <p className="mt-2">Loading courses...</p>
//         </div>
//       ) : (
//         <CourseList
//           courses={courses}
//           totalCount={courses.length} // Pass total count
//         />
//       )}
//     </div>
//   );
// };

// export default CourseSearch;
import React, { useState, useMemo } from "react";
import {
  useGetCoursesQuery,
  filterCourses,
  CourseFilters as CourseFiltersType,
} from "../../services/courses";
import CourseList from "./CourseList";
import CourseFilters from "./CourseFilter";

const CourseSearch: React.FC = () => {
  // Initialize filters state
  const [filters, setFilters] = useState<CourseFiltersType>({
    campus: undefined,
    term: undefined,
    department: undefined,
    courseName: undefined,
    maxCredits: undefined,
    semesters: undefined,
    hideClosedClasses: false,
  });

  // Fetch all courses once - this is the only API call we'll make
  const { data: allCourses = [], isLoading, error } = useGetCoursesQuery();

  // Apply filters to the course data using our utility function
  const filteredCourses = useMemo(() => {
    if (!allCourses) return [];
    return filterCourses(allCourses, filters);
  }, [allCourses, filters]);

  // Handle filter changes
  const handleFilterChange = (newFilters: Partial<CourseFiltersType>) => {
    setFilters((prev) => ({
      ...prev,
      ...newFilters,
    }));
  };

  return (
    <div className="course-search w-full h-full overflow-y-auto bg-white p-6">
      <h1 className="text-2xl font-bold mb-4">Course Search</h1>

      <p className="mb-4">
        Select <strong>Campus</strong> then <strong>Term</strong> to continue
      </p>

      {/* Course Filters */}
      <CourseFilters filters={filters} onFilterChange={handleFilterChange} />

      {/* Loading State */}
      {isLoading && (
        <div className="text-center py-10">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          <p className="mt-2">Loading courses...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-4 bg-red-50 text-red-700 rounded-lg mb-6">
          <h3 className="font-bold">Error Loading Courses</h3>
          <p>
            There was a problem loading the course data. Please try again later.
          </p>
        </div>
      )}

      {/* Results */}
      {!isLoading && !error && (
        <CourseList
          courses={filteredCourses}
          totalCount={filteredCourses.length}
        />
      )}
    </div>
  );
};

export default CourseSearch;
