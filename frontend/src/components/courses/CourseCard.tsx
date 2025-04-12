// components/CourseCard.tsx - For catalog view (first image)
import React from "react";

interface CourseCardProps {
  image: string;
  logo?: string;
  institution: string;
  title: string;
  skills: string[];
  rating: number;
  reviewCount: number;
  level: string;
  type: string;
  duration: string;
  onClick?: () => void;
}

const CourseCard: React.FC<CourseCardProps> = ({
  image,
  logo,
  institution,
  title,
  skills,
  rating,
  reviewCount,
  level,
  type,
  duration,
  onClick,
}) => {
  return (
    <div
      className="bg-white border border-gray-200 rounded-lg overflow-hidden cursor-pointer transform transition-all duration-200 hover:shadow-lg hover:translate-y-[-2px]"
      onClick={onClick}
    >
      {/* Course Image */}
      <div className="relative h-48 overflow-hidden border-b border-gray-100">
        <img src={image} alt={title} className="w-full h-full object-cover" />
      </div>

      {/* Course Content */}
      <div className="p-5">
        {/* Institution with Logo */}
        <div className="flex items-center mb-4">
          {logo && (
            <div className="w-10 h-10 mr-3 flex-shrink-0 border border-gray-200 rounded-md overflow-hidden p-1 bg-white">
              <img
                src={logo}
                alt={institution}
                className="w-full h-full object-contain"
              />
            </div>
          )}
          <p className="text-gray-600 font-medium">{institution}</p>
        </div>

        {/* Course Title */}
        <h3 className="text-xl font-bold text-gray-900 mb-4 leading-tight">
          {title}
        </h3>

        {/* Skills */}
        <div className="mb-5 border-b border-gray-100 pb-4">
          <p className="font-semibold text-gray-700 mb-2">
            Skills you'll gain:
          </p>
          <p className="text-gray-600 line-clamp-2">{skills.join(", ")}</p>
        </div>

        {/* Rating */}
        <div className="flex items-center mb-4">
          <svg
            className="w-6 h-6 text-yellow-400"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118l-2.8-2.034c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path>
          </svg>
          <span className="text-xl font-bold ml-1 mr-2">{rating}</span>
          <span className="text-gray-500">
            · {reviewCount.toLocaleString()} reviews
          </span>
        </div>

        {/* Course Details */}
        <div className="flex flex-wrap text-sm text-gray-600 gap-3">
          <span className="bg-gray-100 px-3 py-1 rounded-full">{level}</span>
          <span className="bg-gray-100 px-3 py-1 rounded-full">{type}</span>
          <span className="bg-gray-100 px-3 py-1 rounded-full">{duration}</span>
        </div>
      </div>
    </div>
  );
};

export default CourseCard;
