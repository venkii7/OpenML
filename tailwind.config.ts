import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx,mdx}",
    "./components/**/*.{ts,tsx}",
    "./content/**/*.{md,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#242424",
        paper: "#fafafa",
        line: "#d7d7d7",
        muted: "rgba(36, 36, 36, 0.52)",
        accent: "#242424",
        signal: "#242424",
        code: "#111827"
      }
    }
  },
  plugins: []
};

export default config;
