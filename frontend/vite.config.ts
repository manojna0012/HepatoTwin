import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dashboard calls /api/...; in development Vite passes those requests to the FastAPI backend
// (uvicorn src.api.app:app on port 8000), so the browser only ever talks to one address.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, rewrite: path => path.replace(/^\/api/, '') },
    },
  },
  preview: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, rewrite: path => path.replace(/^\/api/, '') },
    },
  },
})
