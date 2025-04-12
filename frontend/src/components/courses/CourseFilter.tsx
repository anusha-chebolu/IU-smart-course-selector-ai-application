import React, { useState, useEffect } from "react";
import { CourseFilters as CourseFiltersType } from "../../services/courses";
import { filterOptions } from "../../services/courses";

interface CourseFiltersProps {
  filters: CourseFiltersType;
  onFilterChange: (filters: Partial<CourseFiltersType>) => void;
}

const CourseFilters: React.FC<CourseFiltersProps> = ({
  filters,
  onFilterChange,
}) => {
  const [localCourseName, setLocalCourseName] = useState(
    filters.courseName || ""
  );
  const [isInitialRender, setIsInitialRender] = useState(true);

  // Debounce course name search
  useEffect(() => {
    // Skip the initial render to prevent unnecessary filtering
    if (isInitialRender) {
      setIsInitialRender(false);
      return;
    }

    // Set a timeout to delay filtering
    const timeoutId = setTimeout(() => {
      onFilterChange({ courseName: localCourseName || undefined });
    }, 300);

    // Cleanup function to cancel the timeout
    return () => clearTimeout(timeoutId);
  }, [localCourseName, onFilterChange]);

  // Handle semester selection
  const handleSemesterChange = (semester: string) => {
    const currentSemesters = filters.semesters || [];
    const newSemesters = currentSemesters.includes(semester)
      ? currentSemesters.filter((s) => s !== semester)
      : [...currentSemesters, semester];

    onFilterChange({
      semesters: newSemesters.length > 0 ? newSemesters : undefined,
    });
  };

  // Reset all filters
  const resetFilters = () => {
    onFilterChange({
      campus: undefined,
      term: undefined,
      department: "All",
      courseName: undefined,
      maxCredits: undefined,
      semesters: undefined,
      hideClosedClasses: false,
    });
    setLocalCourseName("");
  };

  // Check if any filters are applied
  const isAnyFilterApplied =
    filters.campus ||
    filters.term ||
    filters.department !== "All" ||
    filters.courseName ||
    filters.maxCredits !== undefined ||
    (filters.semesters && filters.semesters.length > 0) ||
    filters.hideClosedClasses;

  return (
    <div className="course-filters mb-8">
      <div className="mb-4">
        <div className="font-bold text-lg mb-1">
          Select <span className="font-bold">Campus</span> then{" "}
          <span className="font-bold">Term</span> to continue
        </div>
      </div>

      {/* Main Filter Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
        {/* Campus Select */}
        <div>
          <label className="block text-gray-700 mb-2">Campus</label>
          <select
            className="w-full p-2 border border-gray-300 rounded-md outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            value={filters.campus || ""}
            onChange={(e) =>
              onFilterChange({ campus: e.target.value || undefined })
            }
          >
            <option value="">Select Campus</option>
            {filterOptions.campuses
              .filter((c) => c !== "Select Campus")
              .map((campus) => (
                <option key={campus} value={campus}>
                  {campus}
                </option>
              ))}
          </select>
        </div>

        {/* Term Select */}
        <div>
          <label className="block text-gray-700 mb-2">Term</label>
          <select
            className="w-full p-2 border border-blue-500 rounded-md outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            value={filters.term || ""}
            onChange={(e) =>
              onFilterChange({ term: e.target.value || undefined })
            }
          >
            <option value="">Select Term</option>
            {filterOptions.terms
              .filter((t) => t !== "Select Term")
              .map((term) => (
                <option key={term} value={term}>
                  {term}
                </option>
              ))}
          </select>
        </div>

        {/* Department Select */}
        <div>
          <label className="block text-gray-700 mb-2">Department</label>
          <select
            className="w-full p-2 border border-gray-300 rounded-md outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            value={filters.department || "All"}
            onChange={(e) =>
              onFilterChange({
                department: e.target.value === "All" ? "All" : e.target.value,
              })
            }
          >
            <option value="All">All</option>
            {filterOptions.departments
              .filter((dept) => dept !== "All")
              .map((dept) => (
                <option key={dept} value={dept}>
                  {dept}
                </option>
              ))}
          </select>
        </div>

        {/* Max Credits Select */}
        <div>
          <label className="block text-gray-700 mb-2">Max Credits</label>
          <select
            className="w-full p-2 border border-gray-300 rounded-md outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            value={filters.maxCredits?.toString() || "Any"}
            onChange={(e) => {
              const value = e.target.value;
              onFilterChange({
                maxCredits: value === "Any" ? undefined : parseInt(value),
              });
            }}
          >
            <option value="Any">Any</option>
            {filterOptions.credits.map((credit) => (
              <option key={credit} value={credit.toString()}>
                {credit} Credit{credit !== 1 ? "s" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Second Row - Course Search & Semester Options */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Course Name Search */}
        <div>
          <label className="block text-gray-700 mb-2">
            Search by Course Name
          </label>
          <div className="flex">
            <input
              type="text"
              className="flex-grow p-2 border border-gray-300 rounded-l-md outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              placeholder="E.g., BUEX-V 594, Finance, Marketing"
              value={localCourseName}
              onChange={(e) => setLocalCourseName(e.target.value)}
            />
            <button
              type="button"
              className="bg-blue-600 text-white px-6 py-2 rounded-r-md hover:bg-blue-700 transition-colors"
              onClick={() =>
                onFilterChange({ courseName: localCourseName || undefined })
              }
            >
              Search
            </button>
          </div>
        </div>

        {/* Semester Offered */}
        <div>
          <label className="block text-gray-700 mb-2">Semester Offered</label>
          <div className="flex gap-8">
            {["Spring", "Summer", "Fall"].map((semester) => (
              <div key={semester} className="flex items-center">
                <input
                  type="checkbox"
                  id={`semester-${semester}`}
                  checked={(filters.semesters || []).includes(semester)}
                  onChange={() => handleSemesterChange(semester)}
                  className="mr-2 h-4 w-4 text-blue-600 rounded"
                />
                <label
                  htmlFor={`semester-${semester}`}
                  className="text-gray-700"
                >
                  {semester}
                </label>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Hide Closed Classes & Reset Filters */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center">
          <input
            type="checkbox"
            id="hideClosedClasses"
            checked={filters.hideClosedClasses || false}
            onChange={(e) =>
              onFilterChange({ hideClosedClasses: e.target.checked })
            }
            className="mr-2 h-4 w-4 text-blue-600 rounded"
          />
          <label htmlFor="hideClosedClasses" className="text-gray-700">
            Hide closed classes
          </label>
        </div>

        {/* Reset Filters Button - Always show, but disable if no filters applied */}
        <button
          onClick={resetFilters}
          disabled={!isAnyFilterApplied}
          className={`px-6 py-2 rounded-md transition-colors ${
            isAnyFilterApplied
              ? "bg-red-500 text-white hover:bg-red-600"
              : "bg-gray-200 text-gray-500 cursor-not-allowed"
          }`}
        >
          Reset All Filters
        </button>
      </div>
    </div>
  );
};

export default CourseFilters;
