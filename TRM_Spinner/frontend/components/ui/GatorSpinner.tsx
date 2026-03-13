"use client";

interface GatorSpinnerProps {
  size?: number;
  className?: string;
}

export default function GatorSpinner({
  size = 64,
  className = "",
}: GatorSpinnerProps) {
  return (
    <div className={`inline-flex items-center justify-center ${className}`}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        className="gator-spin"
        aria-label="Loading"
      >
        {/* Stylized alligator curled into a circle, chasing its tail */}
        <path
          d="M 50 10
             C 72 10, 90 28, 90 50
             C 90 72, 72 90, 50 90
             C 28 90, 10 72, 10 50
             C 10 34, 20 20, 35 14"
          fill="none"
          stroke="#2d5a3d"
          strokeWidth="6"
          strokeLinecap="round"
        />
        {/* Head - wider snout shape */}
        <path
          d="M 35 14
             L 24 8
             L 28 16
             Z"
          fill="#2d5a3d"
        />
        {/* Eye */}
        <circle cx="33" cy="11" r="2" fill="#f5f0e8" />
        {/* Tail taper at the end */}
        <path
          d="M 50 10
             C 48 10, 46 11, 45 13"
          fill="none"
          stroke="#2d5a3d"
          strokeWidth="3"
          strokeLinecap="round"
        />
        {/* Spine ridges along the body */}
        <circle cx="70" cy="15" r="2" fill="#2d5a3d" />
        <circle cx="85" cy="30" r="2" fill="#2d5a3d" />
        <circle cx="88" cy="50" r="2" fill="#2d5a3d" />
        <circle cx="82" cy="70" r="2" fill="#2d5a3d" />
        <circle cx="68" cy="85" r="2" fill="#2d5a3d" />
        <circle cx="50" cy="88" r="2" fill="#2d5a3d" />
        <circle cx="32" cy="82" r="2" fill="#2d5a3d" />
        <circle cx="18" cy="68" r="2" fill="#2d5a3d" />
        <circle cx="13" cy="50" r="2" fill="#2d5a3d" />
      </svg>
    </div>
  );
}
