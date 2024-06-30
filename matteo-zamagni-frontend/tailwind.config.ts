import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: ["selector"],
  theme: {
    screens: {
      md: "768px",
    },
    extend: {
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      textShadow: {
        sm: "0 1px 2px var(--text-glow-color)",
        DEFAULT: "0 2px 4px var(--text-glow-color)",
        lg: "0 8px 16px var(--text-glow-color)",
      },
      colors: {
        background_Dark: "#000000",
        background_Light: "#A6A7A7",
        text_Dark: "#ffffff",
        text_Light: "#000000",
        ledInactive_Dark: "#333333",
        ledActive_Dark: "#ABABAB",
        ledInactive_Light: "#C5C4C4",
        landingIconInactive: "#545353",
        ledActive_Light: "#000000",
        highlight: "#FEF781",
        fadedWhite: "#cccccc",
        textInactive: "#545353",
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
        arrowGesture: "arrowGesture 5s ease-in-out 2000ms infinite",
        // wiggle: "wiggle 1s ease-in-out infinite",
        pingOnce: "ping 0.5s cubic-bezier(0, 0, 0.2, 1) none",
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
          "100%": { transform: "rotate(0deg)" },
        },
        arrowGesture: {
          "0%": { transform: "translateX(0px)" },
          "8%": { transform: "translateX(4px)" },
          "16%": { transform: "translateX(0px)" },
          "24%": { transform: "translateX(4px)" },
          "32%": { transform: "translateX(0px)" },
          "100%": { transform: "translateX(0px)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
