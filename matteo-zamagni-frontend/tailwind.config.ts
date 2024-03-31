import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: ["selector"],
  theme: {
    extend: {
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      colors: {
        background_Dark: "#000000",
        background_Light: "#A6A7A7",
        ledInactive_Dark: "#333333",
        ledActive_Dark: "#ABABAB",
        ledInactive_Light: "#C5C4C4",
        ledActive_Light: "#000000",
        highlight: "#FEF781",
      },
      fontFamily: {
        mono: ["var(--font-mono)"],
        decoration: ["var(--font-decoration)"],
      },
    },
  },
  plugins: [],
};
export default config;
