/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './templates/**/*.html',
    './static/js/**/*.js'
  ],
  theme: {
    extend: {
      colors: {
        primary: '#0A7CFF',
        secondary: '#0E1A35',
        accent: '#34D399',
        light: '#F8FAFC',
        dark: '#1E293B'
      },
      fontFamily: {
        sans: ['Poppins', 'sans-serif']
      }
    },
  },
  plugins: [],
} 