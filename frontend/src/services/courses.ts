// import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
// import { RootState } from "../store";

// // Hardcoded filter options
// export const filterOptions = {
//   campuses: ["Select Campus", "IU Bloomington"],
//   terms: [
//     "Select Term",
//     "Spring 2023",
//     "Fall 2023",
//     "Spring 2024",
//     "Fall 2024",
//   ],
//   departments: [
//     "All",
//     "AAAD",
//     "AADM",
//     "AAST",
//     "ABEH",
//     "AERO",
//     "AFRI",
//     "AMST",
//     "ANTH",
//     "ARTH",
//     "BUEX",
//     "BUKD",
//     "BUKX",
//     "BUS",
//     "CSCI",
//     "ECON",
//     "EDUC",
//     "INST",
//     "MSCH",
//     "MSCI",
//   ],
//   attributes: ["Online", "In-Person", "Hybrid"],
//   credits: [1, 2, 3, 4, 5, 6, 12],
// };

// // Define filter types to match UI components
// export interface CourseFilters {
//   campus?: string; // "IU Bloomington", etc.
//   term?: string; // Selected term
//   department?: string; // "AAAD", "BUKD", etc. or "All"
//   keyword?: string; // General keyword search
//   courseName?: string; // E.g., "BUEX-V 594, Digital Transformation"
//   maxCredits?: number; // 1, 2, 3, 4, 5, 6, or null for "Any"
//   semesters?: string[]; // Array of selected semesters
//   attributes?: string[]; // Array of selected attributes
//   credits?: string[]; // Credit options as strings
//   level?: string[]; // Course levels
//   instructionMode?: string[]; // In-person, online, etc.
//   hideClosedClasses?: boolean; // Whether to hide closed classes
// }

// // Course types
// export interface Course {
//   course_id: string;
//   course_name: string;
//   department: string;
//   min_credits: number;
//   max_credits: number;
//   prerequisites: string[];
//   offered_semester: string;
//   course_title: string;
//   course_description: string;
//   course_details: {
//     classes: Array<{
//       car: string;
//       moi: string;
//       inst: string;
//       strm: string;
//       campus: string;
//       closed: boolean;
//       acadOrg: string;
//       created: number;
//       primary: boolean;
//       classNbr: number;
//       courseId: string;
//       location: string;
//       maxUnits: number;
//       minUnits: number;
//       classType: string;
//       openSeats: number;
//       totalSeats: number;
//       description: string | null;
//       sessionDescr: string;
//       modeOfInstruction: string;
//       locationDescription: string;
//       primaryInstructors: Array<{
//         role: string;
//         fullName: string;
//         lastName: string;
//         firstName: string;
//         middleName: string | null;
//       }>;
//       meetings: Array<{
//         room: string;
//         meetingDays: string;
//         component: string;
//         buildingName: string;
//         classSection: string;
//       }>;
//     }>;
//     attributes: string[];
//     institution: string;
//     courseTopicId: number;
//     effectiveDate: number;
//     academicCareer: string;
//     courseOfferNumber: number;
//   };
// }

// // Create a separate API for courses
// export const coursesApi = createApi({
//   reducerPath: "coursesApi",
//   baseQuery: fetchBaseQuery({
//     baseUrl: "http://192.168.0.126:8000",
//     prepareHeaders: (headers, { getState }) => {
//       // Get token from Redux store or localStorage
//       const token =
//         (getState() as RootState).auth?.token || localStorage.getItem("token");

//       // If token exists, add it to the headers
//       if (token) {
//         headers.set("Authorization", `Bearer ${token}`);
//       }

//       // Set content type for POST requests
//       if (!headers.has("Content-Type")) {
//         headers.set("Content-Type", "application/json");
//       }

//       return headers;
//     },
//   }),
//   tagTypes: ["Courses", "Coursesfilters"],
//   endpoints: (builder) => ({
//     // Get all courses from external API
//     getCourses: builder.query<Course[], void>({
//       query: () => ({
//         url: "/course_catalog",
//         method: "GET",
//         headers: {
//           Accept: "application/json",
//         },
//       }),
//       // Transform the nested response structure to a flat array of courses
//       transformResponse: (response: any) => {
//         // Check if the response has the nested structure we've observed
//         if (response && response.courses && Array.isArray(response.courses)) {
//           // Handle nested courses array
//           if (
//             response.courses.length > 0 &&
//             response.courses[0].courses &&
//             Array.isArray(response.courses[0].courses)
//           ) {
//             return response.courses[0].courses;
//           }
//           // Handle flat courses array
//           return response.courses;
//         }

//         // If response is already an array of courses, return it
//         if (Array.isArray(response)) {
//           return response;
//         }

//         // Return empty array if no valid format found
//         console.error("Unexpected course data format:", response);
//         return [];
//       },
//       providesTags: ["Courses"],
//     }),

//     // Get filtered courses from external API
//     getFilteredCourses: builder.query<Course[], CourseFilters>({
//       query: (filters) => {
//         // Convert filters to API-compatible params
//         const params: Record<string, string> = {};

//         if (filters.campus && filters.campus !== "Select Campus") {
//           params.campus = filters.campus;
//         }

//         if (filters.term && filters.term !== "Select Term") {
//           params.term = filters.term;
//         }

//         if (filters.department && filters.department !== "All") {
//           params.department = filters.department;
//         }

//         if (filters.courseName) {
//           params.course_name = filters.courseName;
//         }

//         if (filters.keyword) {
//           params.keyword = filters.keyword;
//         }

//         if (filters.maxCredits && filters.maxCredits > 0) {
//           params.max_credits = filters.maxCredits.toString();
//         }

//         // Convert array filters to string params if needed
//         if (filters.semesters && filters.semesters.length > 0) {
//           params.semesters = filters.semesters.join(",");
//         }

//         if (filters.instructionMode && filters.instructionMode.length > 0) {
//           params.instruction_mode = filters.instructionMode.join(",");
//         }

//         if (filters.hideClosedClasses) {
//           params.hide_closed = "true";
//         }

//         return {
//           url: "/course_catalog",
//           method: "GET",
//           params,
//           headers: {
//             Accept: "application/json",
//           },
//         };
//       },
//       // Transform the nested response and apply filters
//       transformResponse: (response: any, _, filters) => {
//         // First extract the courses from the nested structure
//         let courses: Course[] = [];

//         // Check if the response has the nested structure we've observed
//         if (response && response.courses && Array.isArray(response.courses)) {
//           // Handle nested courses array
//           if (
//             response.courses.length > 0 &&
//             response.courses[0].courses &&
//             Array.isArray(response.courses[0].courses)
//           ) {
//             courses = response.courses[0].courses;
//           } else {
//             // Handle flat courses array
//             courses = response.courses;
//           }
//         } else if (Array.isArray(response)) {
//           // If response is already an array of courses, use it
//           courses = response;
//         } else {
//           console.error("Unexpected course data format:", response);
//           return [];
//         }

//         // Now apply client-side filtering if needed
//         if (!filters || Object.keys(filters).length === 0) {
//           return courses;
//         }

//         // Start with all courses
//         let result = [...courses];

//         // Apply campus filter
//         if (filters.campus && filters.campus !== "Select Campus") {
//           result = result.filter((course) =>
//             course.course_details?.classes.some(
//               (cls) => cls.campus === filters.campus
//             )
//           );
//         }

//         // Apply term filter
//         if (filters.term && filters.term !== "Select Term") {
//           result = result.filter((course) =>
//             course.course_details?.classes.some(
//               (cls) => cls.strm === filters.term
//             )
//           );
//         }

//         // Apply department filter
//         if (filters.department && filters.department !== "All") {
//           result = result.filter(
//             (course) => course.department === filters.department
//           );
//         }

//         // Apply keyword filter
//         if (filters.keyword) {
//           const keyword = filters.keyword.toLowerCase();
//           result = result.filter(
//             (course) =>
//               course.course_name.toLowerCase().includes(keyword) ||
//               course.course_title.toLowerCase().includes(keyword) ||
//               course.course_description.toLowerCase().includes(keyword)
//           );
//         }

//         // Apply course name search
//         if (filters.courseName) {
//           const searchTerm = filters.courseName.toLowerCase();
//           result = result.filter(
//             (course) =>
//               course.course_name.toLowerCase().includes(searchTerm) ||
//               course.course_title.toLowerCase().includes(searchTerm)
//           );
//         }

//         // Apply max credits filter (using specific UI values)
//         if (filters.maxCredits && filters.maxCredits > 0) {
//           result = result.filter(
//             (course) => course.max_credits <= filters.maxCredits!
//           );
//         }

//         // Apply instruction mode filters
//         if (filters.instructionMode && filters.instructionMode.length > 0) {
//           result = result.filter((course) =>
//             course.course_details?.classes.some((cls) =>
//               filters.instructionMode!.includes(cls.modeOfInstruction)
//             )
//           );
//         }

//         // Apply hide closed classes filter
//         if (filters.hideClosedClasses) {
//           result = result.filter((course) =>
//             course.course_details?.classes.some((cls) => !cls.closed)
//           );
//         }

//         // Apply semester filters
//         if (filters.semesters && filters.semesters.length > 0) {
//           result = result.filter((course) => {
//             return filters.semesters!.some((semester) =>
//               course.offered_semester.includes(semester)
//             );
//           });
//         }

//         return result;
//       },
//       providesTags: ["Courses"],
//     }),

//     // Get course by ID from external API
//     getCourseById: builder.query<Course, string>({
//       query: (courseId) => ({
//         url: `/course_catalog/${courseId}`,
//         method: "GET",
//         headers: {
//           Accept: "application/json",
//         },
//       }),
//       // Transform response in case it comes in the nested format
//       transformResponse: (response: any) => {
//         // If the response is directly a course object
//         if (response && response.course_id) {
//           return response;
//         }

//         // If the response has a nested structure
//         if (response && response.course) {
//           return response.course;
//         }

//         // If we can't find a valid course, log an error and return null
//         console.error("Unexpected course data format:", response);
//         return null;
//       },
//       providesTags: (result, error, id) => [{ type: "Courses", id }],
//     }),

//     // Get filter options - now returns hardcoded values instead of making an API call
//     getFilterOptions: builder.query<
//       {
//         campuses: string[];
//         terms: string[];
//         departments: string[];
//         attributes: string[];
//         credits: number[];
//       },
//       void
//     >({
//       // This is still a query, but we'll make it synchronous by
//       // immediately returning the hardcoded options
//       queryFn: () => {
//         return {
//           data: filterOptions,
//         };
//       },
//       providesTags: ["Coursesfilters"],
//     }),
//   }),
// });

// // Export hooks for the courses API
// export const {
//   useGetCoursesQuery,
//   useGetFilteredCoursesQuery,
//   useGetCourseByIdQuery,
//   useGetFilterOptionsQuery,
// } = coursesApi;

import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { RootState } from "../store";

// Hardcoded filter options - export directly for component usage
export const filterOptions = {
  campuses: ["Select Campus", "IU Bloomington"],
  terms: [
    "Select Term",
    "Spring 2023",
    "Fall 2023",
    "Spring 2024",
    "Fall 2024",
  ],
  departments: [
    "All",
    "AAAD",
    "AADM",
    "AAST",
    "ABEH",
    "AERO",
    "AFRI",
    "AMST",
    "ANTH",
    "ARTH",
    "BUEX",
    "BUKD",
    "BUKX",
    "BUS",
    "CSCI",
    "ECON",
    "EDUC",
    "INST",
    "MSCH",
    "MSCI",
  ],
  attributes: ["Online", "In-Person", "Hybrid"],
  credits: [1, 2, 3, 4, 5, 6, 12],
};

// Define filter types to match UI components
export interface CourseFilters {
  campus?: string;
  term?: string;
  department?: string;
  keyword?: string;
  courseName?: string;
  maxCredits?: number;
  semesters?: string[];
  attributes?: string[];
  credits?: string[];
  level?: string[];
  instructionMode?: string[];
  hideClosedClasses?: boolean;
}

// Course types
export interface Course {
  course_id: string;
  course_name: string;
  department: string;
  min_credits: number;
  max_credits: number;
  prerequisites: string[];
  offered_semester: string;
  course_title: string;
  course_description: string;
  course_details: {
    classes: Array<{
      car: string;
      moi: string;
      inst: string;
      strm: string;
      campus: string;
      closed: boolean;
      acadOrg: string;
      created: number;
      primary: boolean;
      classNbr: number;
      courseId: string;
      location: string;
      maxUnits: number;
      minUnits: number;
      classType: string;
      openSeats: number;
      totalSeats: number;
      description: string | null;
      sessionDescr: string;
      modeOfInstruction: string;
      locationDescription: string;
      primaryInstructors: Array<{
        role: string;
        fullName: string;
        lastName: string;
        firstName: string;
        middleName: string | null;
      }>;
      meetings: Array<{
        room: string;
        meetingDays: string;
        component: string;
        buildingName: string;
        classSection: string;
      }>;
    }>;
    attributes: string[];
    institution: string;
    courseTopicId: number;
    effectiveDate: number;
    academicCareer: string;
    courseOfferNumber: number;
  };
}

// Function to filter courses on the frontend
export function filterCourses(
  courses: Course[],
  filters: CourseFilters
): Course[] {
  if (!courses || !filters || Object.keys(filters).length === 0) {
    return courses || [];
  }

  // Start with all courses
  let result = [...courses];

  // Apply campus filter
  if (filters.campus && filters.campus !== "Select Campus") {
    result = result.filter((course) =>
      course.course_details?.classes.some(
        (cls) => cls.campus === filters.campus
      )
    );
  }

  // Apply term filter
  if (filters.term && filters.term !== "Select Term") {
    result = result.filter((course) =>
      course.course_details?.classes.some((cls) => cls.strm === filters.term)
    );
  }

  // Apply department filter
  if (filters.department && filters.department !== "All") {
    result = result.filter(
      (course) => course.department === filters.department
    );
  }

  // Apply keyword filter
  if (filters.keyword) {
    const keyword = filters.keyword.toLowerCase();
    result = result.filter(
      (course) =>
        course.course_name.toLowerCase().includes(keyword) ||
        course.course_title.toLowerCase().includes(keyword) ||
        course.course_description.toLowerCase().includes(keyword)
    );
  }

  // Apply course name search
  if (filters.courseName) {
    const searchTerm = filters.courseName.toLowerCase();
    result = result.filter(
      (course) =>
        course.course_name.toLowerCase().includes(searchTerm) ||
        course.course_title.toLowerCase().includes(searchTerm)
    );
  }

  // Apply max credits filter (using specific UI values)
  if (filters.maxCredits && filters.maxCredits > 0) {
    result = result.filter(
      (course) => course.max_credits <= filters.maxCredits!
    );
  }

  // Apply instruction mode filters
  if (filters.instructionMode && filters.instructionMode.length > 0) {
    result = result.filter((course) =>
      course.course_details?.classes.some((cls) =>
        filters.instructionMode!.includes(cls.modeOfInstruction)
      )
    );
  }

  // Apply hide closed classes filter
  if (filters.hideClosedClasses) {
    result = result.filter((course) =>
      course.course_details?.classes.some((cls) => !cls.closed)
    );
  }

  // Apply semester filters
  if (filters.semesters && filters.semesters.length > 0) {
    result = result.filter((course) => {
      return filters.semesters!.some((semester) =>
        course.offered_semester.includes(semester)
      );
    });
  }

  return result;
}

// Helper function to get a course by ID from the array (instead of API call)
export function getCourseById(
  courses: Course[],
  courseId: string
): Course | undefined {
  return courses.find((course) => course.course_id === courseId);
}

// Create a separate API for courses
export const coursesApi = createApi({
  reducerPath: "coursesApi",
  baseQuery: fetchBaseQuery({
    baseUrl: "http://192.168.0.126:8000",
    prepareHeaders: (headers, { getState }) => {
      // Get token from Redux store or localStorage
      const token =
        (getState() as RootState).auth?.token || localStorage.getItem("token");

      // If token exists, add it to the headers
      if (token) {
        headers.set("Authorization", `Bearer ${token}`);
      }

      // Set content type for POST requests
      if (!headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
      }

      return headers;
    },
  }),
  tagTypes: ["Courses"],
  endpoints: (builder) => ({
    // Get all courses from external API - this is the only endpoint we'll use
    getCourses: builder.query<Course[], void>({
      query: () => ({
        url: "/course_catalog",
        method: "GET",
        headers: {
          Accept: "application/json",
        },
      }),
      // Transform the nested response structure to a flat array of courses
      transformResponse: (response: any) => {
        // Check if the response has the nested structure we've observed
        if (response && response.courses && Array.isArray(response.courses)) {
          // Handle nested courses array
          if (
            response.courses.length > 0 &&
            response.courses[0].courses &&
            Array.isArray(response.courses[0].courses)
          ) {
            return response.courses[0].courses;
          }
          // Handle flat courses array
          return response.courses;
        }

        // If response is already an array of courses, return it
        if (Array.isArray(response)) {
          return response;
        }

        // Return empty array if no valid format found
        console.error("Unexpected course data format:", response);
        return [];
      },
      providesTags: ["Courses"],
    }),
  }),
});

// Export hooks for the courses API
export const { useGetCoursesQuery } = coursesApi;

// Course Trends Interface
export interface CourseTrend {
  year: number;
  slots_filled: number;
  total_slots: number;
  avg_rating: number;
  slots_filled_time: number;
  avg_gpa: number;
  avg_hours_spent: number;
}

export interface CourseTrendsResponse {
  course_id: string;
  course_name: string;
  trends: CourseTrend[];
  trend_analysis: string;
}

export const courseTrendsApi = createApi({
  reducerPath: "courseTrendsApi",
  baseQuery: fetchBaseQuery({
    baseUrl: "http://192.168.0.126:8000",
    prepareHeaders: (headers, { getState }) => {
      // Get token from Redux store or localStorage
      const token =
        (getState() as RootState).auth?.token || localStorage.getItem("token");

      // If token exists, add it to the headers
      if (token) {
        headers.set("Authorization", `Bearer ${token}`);
      }

      // Set content type for POST requests
      if (!headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
      }

      return headers;
    },
  }),
  tagTypes: ["CourseTrends"],
  endpoints: (builder) => ({
    // Get course trends by course ID
    getCourseTrends: builder.query<CourseTrendsResponse, string>({
      query: (courseId) => ({
        url: `/trends/${courseId}`,
        method: "GET",
        headers: {
          Accept: "application/json",
        },
      }),
      providesTags: (result, error, courseId) => [
        { type: "CourseTrends", id: courseId },
      ],
    }),
  }),
});

// Export hooks for the course trends API
export const { useGetCourseTrendsQuery } = courseTrendsApi;
