// import React from "react";
// import { Course } from "../../types";

// interface CourseItemProps {
//   key: string;
//   course: Course;
//   isExpanded: boolean;
//   onToggleExpand: () => void;
// }

// const CourseItem: React.FC<CourseItemProps> = ({
//   key,
//   course,
//   isExpanded,
//   onToggleExpand,
// }) => {
//   return (
//     <div
//       key={key}
//       className="course-item border border-gray-200 rounded-md overflow-hidden bg-white mb-px animate-fade-in"
//     >
//       {/* Course Header - Always visible */}
//       <div
//         className="course-header flex cursor-pointer items-center justify-center"
//         onClick={onToggleExpand}
//       >
//         {/* Left column with course code and credits */}
//         <div className="course-code-section w-[200px] min-w-[200px] p-4 bg-gray-50 border-r border-gray-200 text-center">
//           <h3 className="course-code font-bold text-gray-800">
//             {course.course_name}
//           </h3>
//           <p className="course-credits text-sm text-gray-600">
//             {course.max_credits} credit{course.max_credits !== 1 ? "s" : ""}
//           </p>
//         </div>

//         {/* Right column with course title */}
//         <div className="course-title-section flex-grow p-4 flex items-center justify-between">
//           <div className="flex-grow text-center">
//             <h2 className="course-title font-bold text-gray-900">
//               {course.course_title}
//             </h2>
//           </div>

//           <div>
//             <button
//               className="expand-button"
//               aria-label={isExpanded ? "Collapse" : "Expand"}
//             >
//               <svg
//                 className={`w-6 h-6 transform transition-transform ${
//                   isExpanded ? "rotate-180" : ""
//                 }`}
//                 fill="none"
//                 stroke="currentColor"
//                 viewBox="0 0 24 24"
//               >
//                 <path
//                   strokeLinecap="round"
//                   strokeLinejoin="round"
//                   strokeWidth={2}
//                   d="M19 9l-7 7-7-7"
//                 />
//               </svg>
//             </button>
//           </div>
//         </div>
//       </div>

//       {/* Expanded Content */}
//       {isExpanded && (
//         <div className="expanded-content border-t border-gray-200 bg-white animate-fade-in">
//           <div className="grid grid-cols-1 lg:grid-cols-2">
//             <div className="p-4 border-b lg:border-b-0 lg:border-r border-gray-200">
//               <div className="mb-4">
//                 <h3 className="font-semibold text-gray-800 mb-1">
//                   Typically Offered:
//                 </h3>
//                 <p>{course.offered_semester || "Not specified"}</p>
//               </div>

//               <div>
//                 <h3 className="font-semibold text-gray-800 mb-1">
//                   Course Description:
//                 </h3>
//                 <p className="text-gray-700">{course.course_description}</p>
//               </div>
//             </div>

//             <div className="p-4">
//               <div className="mb-4">
//                 <h3 className="font-semibold text-gray-800 mb-1">
//                   Instructors:
//                 </h3>
//                 <ul className="list-disc list-inside">
//                   {course.course_details?.classes?.[0]?.primaryInstructors?.map(
//                     (instructor, index) => (
//                       <li key={index}>{instructor.fullName}</li>
//                     )
//                   ) || <li>No instructors listed</li>}
//                 </ul>
//               </div>

//               <div>
//                 <h3 className="font-semibold text-gray-800 mb-1">
//                   Class Information:
//                 </h3>
//                 <div className="grid grid-cols-2 gap-4 bg-gray-50 p-4 rounded-md">
//                   <div>
//                     <p className="text-xs text-gray-500 uppercase">
//                       Mode of Instruction:
//                     </p>
//                     <p>
//                       {course.course_details?.classes?.[0]?.modeOfInstruction ||
//                         "Not specified"}
//                     </p>
//                   </div>
//                   <div>
//                     <p className="text-xs text-gray-500 uppercase">Location:</p>
//                     <p>
//                       {course.course_details?.classes?.[0]
//                         ?.locationDescription || "Not specified"}
//                     </p>
//                   </div>
//                   <div>
//                     <p className="text-xs text-gray-500 uppercase">
//                       Available Seats:
//                     </p>
//                     <p>
//                       {course.course_details?.classes?.[0]?.openSeats || 0} of{" "}
//                       {course.course_details?.classes?.[0]?.totalSeats || 0}
//                     </p>
//                   </div>
//                   <div>
//                     <p className="text-xs text-gray-500 uppercase">Session:</p>
//                     <p>
//                       {course.course_details?.classes?.[0]?.sessionDescr ||
//                         "Standard Session"}
//                     </p>
//                   </div>
//                 </div>
//               </div>
//             </div>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default CourseItem;
import React, { useState } from "react";
import { Course } from "../../types";
import { useGetCourseTrendsQuery } from "../../services/courses";
import { Line } from "recharts";
import {
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface CourseItemProps {
  key: string;
  course: Course;
  isExpanded: boolean;
  onToggleExpand: () => void;
}

const CourseItem: React.FC<CourseItemProps> = ({
  key,
  course,
  isExpanded,
  onToggleExpand,
}) => {
  const [showTrends, setShowTrends] = useState(false);
  const { data: trendsData, isLoading: isTrendsLoading } =
    useGetCourseTrendsQuery(course.course_id, {
      skip: !showTrends, // Only fetch when trends are requested
    });

  // Toggle trends visibility
  const handleShowTrends = () => {
    setShowTrends(!showTrends);
  };

  // Render trends chart
  const renderTrendsChart = () => {
    if (!showTrends) return null;

    if (isTrendsLoading) {
      return (
        <div className="trends-loading flex justify-center items-center p-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        </div>
      );
    }

    if (!trendsData) return null;

    return (
      <div className="trends-section mt-4 bg-gray-50 p-4 rounded-md">
        <h3 className="font-semibold text-gray-800 mb-3">
          Course Performance Trends
        </h3>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendsData.trends}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="avg_rating"
              stroke="#8884d8"
              activeDot={{ r: 8 }}
            />
            <Line type="monotone" dataKey="avg_gpa" stroke="#82ca9d" />
            <Line type="monotone" dataKey="avg_hours_spent" stroke="#ffc658" />
          </LineChart>
        </ResponsiveContainer>

        {/* Trend Analysis */}
        <div className="trend-analysis mt-4 bg-white p-3 rounded-md">
          <h4 className="font-medium text-gray-700 mb-2">Trend Analysis</h4>
          <p className="text-sm text-gray-600 whitespace-pre-line">
            {trendsData.trend_analysis}
          </p>
        </div>
      </div>
    );
  };

  return (
    <div
      key={key}
      className="course-item border border-gray-200 rounded-md overflow-hidden bg-white mb-px animate-fade-in"
    >
      {/* Course Header - Always visible */}
      <div
        className="course-header flex cursor-pointer items-center justify-center"
        onClick={onToggleExpand}
      >
        {/* Left column with course code and credits */}
        <div className="course-code-section w-[200px] min-w-[200px] p-4 bg-gray-50 border-r border-gray-200 text-center">
          <h3 className="course-code font-bold text-gray-800">
            {course.course_name}
          </h3>
          <p className="course-credits text-sm text-gray-600">
            {course.max_credits} credit{course.max_credits !== 1 ? "s" : ""}
          </p>
        </div>

        {/* Right column with course title */}
        <div className="course-title-section flex-grow p-4 flex items-center justify-between">
          <div className="flex-grow text-center">
            <h2 className="course-title font-bold text-gray-900">
              {course.course_title}
            </h2>
          </div>

          <div>
            <button
              className="expand-button"
              aria-label={isExpanded ? "Collapse" : "Expand"}
            >
              <svg
                className={`w-6 h-6 transform transition-transform ${
                  isExpanded ? "rotate-180" : ""
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="expanded-content border-t border-gray-200 bg-white animate-fade-in">
          <div className="grid grid-cols-1 lg:grid-cols-2">
            <div className="p-4 border-b lg:border-b-0 lg:border-r border-gray-200">
              <div className="mb-4">
                <h3 className="font-semibold text-gray-800 mb-1">
                  Typically Offered:
                </h3>
                <p>{course.offered_semester || "Not specified"}</p>
              </div>

              <div>
                <h3 className="font-semibold text-gray-800 mb-1">
                  Course Description:
                </h3>
                <p className="text-gray-700">{course.course_description}</p>
              </div>
            </div>

            <div className="p-4">
              <div className="mb-4">
                <h3 className="font-semibold text-gray-800 mb-1">
                  Instructors:
                </h3>
                <ul className="list-disc list-inside">
                  {course.course_details?.classes?.[0]?.primaryInstructors?.map(
                    (instructor, index) => (
                      <li key={index}>{instructor.fullName}</li>
                    )
                  ) || <li>No instructors listed</li>}
                </ul>
              </div>

              <div>
                <h3 className="font-semibold text-gray-800 mb-1">
                  Class Information:
                </h3>
                <div className="grid grid-cols-2 gap-4 bg-gray-50 p-4 rounded-md">
                  <div>
                    <p className="text-xs text-gray-500 uppercase">
                      Mode of Instruction:
                    </p>
                    <p>
                      {course.course_details?.classes?.[0]?.modeOfInstruction ||
                        "Not specified"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 uppercase">Location:</p>
                    <p>
                      {course.course_details?.classes?.[0]
                        ?.locationDescription || "Not specified"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 uppercase">
                      Available Seats:
                    </p>
                    <p>
                      {course.course_details?.classes?.[0]?.openSeats || 0} of{" "}
                      {course.course_details?.classes?.[0]?.totalSeats || 0}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 uppercase">Session:</p>
                    <p>
                      {course.course_details?.classes?.[0]?.sessionDescr ||
                        "Standard Session"}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Show Trends Button */}
          <div className="p-4 border-t border-gray-200 flex justify-center">
            <button
              onClick={handleShowTrends}
              className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors"
            >
              {showTrends ? "Hide Trends" : "Show Course Trends"}
            </button>
          </div>

          {/* Trends Section */}
          {renderTrendsChart()}
        </div>
      )}
    </div>
  );
};

export default CourseItem;
