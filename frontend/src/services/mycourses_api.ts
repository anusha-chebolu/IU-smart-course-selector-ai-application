import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import {
  QueryReturnValue,
  FetchBaseQueryError,
  FetchBaseQueryMeta,
} from "@reduxjs/toolkit/query";
import { faker } from "@faker-js/faker";

import { Course } from "./courses";
export interface MyCourse {
  id: string;
  userId: string;
  courseId: string;
  addedAt?: string;
}

// Updated type to resolve TypeScript issues
export interface AddCoursePayload {
  courseId: string;
  course?: Partial<Course>;
  overrideExisting?: boolean;
}

export const myCourseApi = createApi({
  reducerPath: "myCourseApi",
  baseQuery: fetchBaseQuery({
    baseUrl: process.env.REACT_APP_API_BASE_URL || "http://192.168.0.139:3001",
    prepareHeaders: (headers, { getState }) => {
      // Get user from localStorage
      const user = localStorage.getItem("user");
      const parsedUser = user ? JSON.parse(user) : null;

      // If user exists, add user ID to the headers
      if (parsedUser && parsedUser.id) {
        headers.set("X-User-Id", parsedUser.id);
      }

      // Set content type for POST requests
      if (!headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
      }

      return headers;
    },
  }),
  tagTypes: ["MyCourses", "Courses"],
  endpoints: (builder) => ({
    // Add a course to user's courses
    addCourseToMyCourses: builder.mutation<MyCourse, AddCoursePayload>({
      queryFn: async (
        { courseId, overrideExisting = false },
        _queryApi,
        _extraOptions,
        fetchWithBQ
      ): Promise<
        QueryReturnValue<MyCourse, FetchBaseQueryError, FetchBaseQueryMeta>
      > => {
        const user = localStorage.getItem("user");
        const parsedUser = user ? JSON.parse(user) : null;

        if (!parsedUser?.id) {
          return {
            error: {
              status: 400,
              data: "User ID not found",
            } as FetchBaseQueryError,
          };
        }

        try {
          // Check if the course exists in external API first
          const courseResponse = await fetch(
            `http://192.168.0.126:8000/course_detail/${courseId}`
          );

          if (!courseResponse.ok) {
            return {
              error: {
                status: 404,
                data: `Course with ID ${courseId} not found in external API`,
              } as FetchBaseQueryError,
            };
          }

          // Check if the course is already in user's courses
          const myCoursesResponse = await fetchWithBQ({
            url: `/my_courses?userId=${parsedUser.id}&courseId=${courseId}`,
            method: "GET",
          });

          if ("error" in myCoursesResponse) {
            throw myCoursesResponse.error;
          }

          const existingMyCourses = myCoursesResponse.data as MyCourse[];

          // If course is not already in user's courses, or we want to override
          if (existingMyCourses.length === 0 || overrideExisting) {
            const newMyCourse: MyCourse = {
              id: faker.string.uuid(),
              userId: parsedUser.id,
              courseId: courseId,
              addedAt: new Date().toISOString(),
            };

            const addMyCourseResponse = await fetchWithBQ({
              url: "/my_courses",
              method: "POST",
              body: newMyCourse,
            });

            if ("error" in addMyCourseResponse) {
              throw addMyCourseResponse.error;
            }

            return {
              data: addMyCourseResponse.data as MyCourse,
            };
          } else {
            // Course already exists in user's courses
            return {
              data: existingMyCourses[0],
            };
          }
        } catch (error) {
          return {
            error: {
              status: 500,
              data: `Failed to add course: ${
                error instanceof Error ? error.message : String(error)
              }`,
            } as FetchBaseQueryError,
          };
        }
      },
      invalidatesTags: ["MyCourses"],
    }),

    // Remove a course from user's courses
    removeCourseFromMyCourses: builder.mutation<void, string>({
      query: (myCourseId) => ({
        url: `/my_courses/${myCourseId}`,
        method: "DELETE",
      }),
      invalidatesTags: ["MyCourses"],
    }),

    // Get user's courses with full course details
    getUserCourses: builder.query<
      (MyCourse & { courseDetails: Course })[],
      void
    >({
      async queryFn(_arg, _queryApi, _extraOptions, fetchWithBQ) {
        const user = localStorage.getItem("user");
        const parsedUser = user ? JSON.parse(user) : null;

        if (!parsedUser?.id) {
          return {
            error: {
              status: 400,
              data: "User ID not found",
            } as FetchBaseQueryError,
          };
        }

        try {
          // Get user's my_courses
          const myCoursesResponse = await fetchWithBQ({
            url: `/my_courses?userId=${parsedUser.id}`,
            method: "GET",
          });

          if ("error" in myCoursesResponse) {
            throw myCoursesResponse.error;
          }

          const myCourses = myCoursesResponse.data as MyCourse[];

          // Fetch details for each course from external API
          const coursesWithDetails = await Promise.all(
            myCourses.map(async (myCourse) => {
              const courseResponse = await fetch(
                `http://192.168.0.126:8000/course_detail/${myCourse.courseId}`
              );

              if (!courseResponse.ok) {
                throw new Error(
                  `Failed to fetch course details for ${myCourse.courseId}`
                );
              }

              const courseDetails = (await courseResponse.json()) as Course;
              return {
                ...myCourse,
                courseDetails,
              };
            })
          );

          return { data: coursesWithDetails };
        } catch (error) {
          return {
            error: {
              status: 500,
              data: `Failed to get user courses: ${
                error instanceof Error ? error.message : String(error)
              }`,
            } as FetchBaseQueryError,
          };
        }
      },
      providesTags: ["MyCourses"],
    }),

    // Get course details
    getCourseDetails: builder.query<Course, string>({
      queryFn: async (courseId) => {
        try {
          const response = await fetch(
            `http://192.168.0.126:8000/course_detail/${courseId}`
          );

          if (!response.ok) {
            throw new Error(`Failed to fetch course details for ${courseId}`);
          }

          const courseDetails = await response.json();
          return { data: courseDetails };
        } catch (error) {
          return {
            error: {
              status: 500,
              data: `Failed to get course details: ${
                error instanceof Error ? error.message : String(error)
              }`,
            } as FetchBaseQueryError,
          };
        }
      },
      providesTags: (result, error, courseId) => [
        { type: "Courses", id: courseId },
      ],
    }),

    // Search courses
    searchCourses: builder.query<Course[], string>({
      query: (searchTerm) => `/courses?q=${encodeURIComponent(searchTerm)}`,
      providesTags: ["Courses"],
    }),
  }),
});

// Export hooks for using the API
export const {
  useAddCourseToMyCoursesMutation,
  useRemoveCourseFromMyCoursesMutation,
  useGetUserCoursesQuery,
  useGetCourseDetailsQuery,
  useSearchCoursesQuery,
} = myCourseApi;
