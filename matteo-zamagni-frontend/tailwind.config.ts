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
        landingIconInactive: "#545353",
        ledActive_Light: "#000000",
        highlight: "#FEF781",
        fadedWhite: "#cccccc",
      },
      fontFamily: {
        mono: ["var(--font-mono)"],
        decoration: ["var(--font-decoration)"],
      },
      animation: {
        downTriangle_TopLine_Forwards:
          "downTriangle_TopLine_Forwards 0.5s ease-in-out forwards",
        downTriangle_LeftLine_Forwards:
          "downTriangle_LeftLine_Forwards 0.5s ease-in-out forwards",
        downTriangle_RightLine_Forwards:
          "downTriangle_RightLine_Forwards 0.5s ease-in-out forwards",
        downTriangle_TopLine_Reverse:
          "downTriangle_TopLine_Reverse 0.5s ease-in-out forwards",
        downTriangle_LeftLine_Reverse:
          "downTriangle_LeftLine_Reverse 0.5s ease-in-out forwards",
        downTriangle_RightLine_Reverse:
          "downTriangle_RightLine_Reverse 0.5s ease-in-out forwards",
        // wiggle: "wiggle 1s ease-in-out infinite",
      },
      keyframes: {
        downTriangle_TopLine_Forwards: {
          "0%": { transform: "rotate(0deg)", opacity: "1" },
          "32%": { opacity: "1" },
          "33%": { transform: "rotate(60deg)", opacity: "0" },
          "100%": { transform: "rotate(60deg)", opacity: "0" },
        },
        downTriangle_LeftLine_Forwards: {
          "0%": { transform: "rotate(0deg)" },
          "33%": { transform: "rotate(0deg)" },
          "100%": { transform: "translate(-28px, -15px) rotate(75deg)" },
        },
        downTriangle_RightLine_Forwards: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "translate(28px, -15px) rotate(-75deg)" },
        },
        downTriangle_TopLine_Reverse: {
          "0%": { transform: "rotate(60deg)", opacity: "0" },
          "32%": { transform: "rotate(60deg)", opacity: "0" },
          "33%": { transform: "rotate(60deg)", opacity: "1" },
          "100%": { transform: "rotate(0deg)", opacity: "1" },
        },
        downTriangle_LeftLine_Reverse: {
          "0%": { transform: "translate(-28px, -15px) rotate(75deg)" },
          "33%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(0deg)" },
        },
        downTriangle_RightLine_Reverse: {
          "0%": { transform: "translate(28px, -15px) rotate(-75deg)" },
          "1000%": { transform: "rotate(0deg)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
