import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        gator: {
          50: "#f0f7f0",
          100: "#e8f5e9",
          200: "#c8e6c9",
          500: "#2d5a3d",
          600: "#1a3a2a",
          700: "#0f2518",
        },
        cream: "#f5f0e8",
        "amber-accent": "#d4a574",
      },
      fontFamily: {
        heading: ["var(--font-caveat)", "cursive"],
        body: ["var(--font-patrick-hand)", "cursive"],
      },
    },
  },
  plugins: [],
};

export default config;
