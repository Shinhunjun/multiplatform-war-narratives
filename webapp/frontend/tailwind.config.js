/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'background-primary': '#1a1d27',
        'background-secondary': '#2a2e3d',
        'text-primary': '#e8eaed',
        'text-secondary': '#8b8fa3',
        'text-muted': '#64748b',
        'accent-reddit': '#6366f1',
        'accent-news': '#f59e0b',
        'accent-tiktok': '#ff0050',
        'sentiment-positive': '#34d399',
        'sentiment-neutral': '#64748b',
        'sentiment-negative': '#f87171',
      }
    },
  },
  plugins: [],
}
