import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      typography: {
        DEFAULT: {
          css: {
            maxWidth: "none",
            color: "#e5e7eb",
            h1: { color: "#f9fafb" },
            h2: { color: "#f3f4f6" },
            h3: { color: "#f3f4f6" },
            strong: { color: "#f9fafb" },
            a: { color: "#818cf8" },
            code: { color: "#a5b4fc", backgroundColor: "#1e1b4b", borderRadius: "4px", padding: "2px 4px" },
            blockquote: { color: "#9ca3af", borderLeftColor: "#4f46e5" },
          },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
