/**
 * API Configuration
 * Supports both local development and deployed environments
 */

const getAPIUrl = () => {
  // If VITE_API_URL is set (from Render or other deployment), use it
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  
  // Local development
  return "http://127.0.0.1:8000";
};

export const API_BASE_URL = getAPIUrl();

console.log("API Base URL:", API_BASE_URL);
