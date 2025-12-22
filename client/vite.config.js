import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    base: '/',
    server: {
        port: 3000,
        proxy: {
            '/api': {
                target: 'https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net',
                changeOrigin: true,
                secure: true,
            },
        },
    },
    build: {
        rollupOptions: {
            output: {
                manualChunks: undefined,
            },
        },
    },
    publicDir: 'public',
})
