# Image Deblurring Application - React + Node.js

A modern web application for AI-powered image deblurring using deep learning and Azure cloud services.

## 🏗️ Project Structure

```
image_deblurring/
├── client/              # React frontend (Vite + Tailwind CSS)
│   ├── src/
│   │   ├── App.jsx     # Main application component
│   │   ├── main.jsx    # React entry point
│   │   └── index.css   # Global styles
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── server/              # Node.js Express backend
│   ├── index.js        # Express server with Azure Function proxy
│   ├── package.json
│   └── .env            # Environment variables
│
├── function_app/        # Azure Function (Python)
├── src/                 # ML training code
└── package.json         # Root package.json with helper scripts
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+ (for Azure Function development)

### Installation

1. **Install all dependencies:**
   ```bash
   npm run install:all
   ```

   Or install manually:
   ```bash
   # Install root dependencies
   npm install

   # Install client dependencies
   cd client && npm install && cd ..

   # Install server dependencies
   cd server && npm install && cd ..
   ```

### Development

2. **Start both frontend and backend:**
   ```bash
   npm run dev
   ```

   This will start:
   - React frontend on `http://localhost:3000`
   - Express backend on `http://localhost:5000`

   Or run them separately:
   ```bash
   # Terminal 1 - Backend
   npm run dev:server

   # Terminal 2 - Frontend
   npm run dev:client
   ```

3. **Open your browser:**
   Navigate to `http://localhost:3000`

## 🎨 Features

- **Modern React UI** with Tailwind CSS
- **Drag & Drop** image upload
- **Before/After Comparison Slider**
- **Retro Terminal** animations during processing (3 themes!)
- **Responsive Design** - works on mobile and desktop
- **Azure Function Integration** for serverless ML inference
- **Real-time Progress** tracking

## 🛠️ Tech Stack

### Frontend
- **React 18** - UI library
- **Vite** - Build tool (fast!)
- **Tailwind CSS** - Utility-first CSS
- **Lucide React** - Beautiful icons

### Backend
- **Express** - Node.js web framework
- **Multer** - File upload handling
- **CORS** - Cross-origin requests
- **Dotenv** - Environment configuration

### ML/Cloud
- **PyTorch** - Deep learning framework
- **Azure Functions** - Serverless compute
- **Custom U-Net** - CNN architecture for deblurring

## 📝 Configuration

### Server Configuration
Edit `server/.env` to configure:
```env
PORT=5000
AZURE_FUNCTION_URL=https://your-function.azurewebsites.net/api/imagedeblur
```

### Client Configuration
The Vite dev server is configured to proxy API requests to the backend:
```javascript
// client/vite.config.js
proxy: {
  '/api': {
    target: 'http://localhost:5000',
    changeOrigin: true,
  },
}
```

## 📦 Production Build

Build the React app for production:
```bash
npm run build
```

The optimized files will be in `client/dist/`.

Preview the production build:
```bash
npm run preview
```

## 🧪 API Endpoints

### Backend (Express)
- `GET /api/health` - Health check
- `POST /api/deblur` - Image deblurring endpoint
  - Accepts: `multipart/form-data` with `image` field
  - Returns: JSON with `deblurred_image` (base64)

### Azure Function
- `POST /api/imagedeblur` - Direct Azure Function endpoint
  - Accepts: JSON with `image` field (base64 data URI)
  - Returns: JSON with deblurred image

## 🎯 Usage

1. Upload a blurred image (drag & drop or click to browse)
2. Click "Deblur Image"
3. Watch the retro terminal animation (try different themes!)
4. Compare before/after with the interactive slider
5. Download your enhanced image

## 🐛 Troubleshooting

**Port already in use:**
```bash
# Change port in server/.env
PORT=5001
```

**Azure Function timeout:**
- Images larger than 1920×1080 may timeout
- Try resizing before upload

**Module not found:**
```bash
npm run install:all
```

## 📄 License

MIT

## 👨‍💻 Author

Built as a portfolio project demonstrating full-stack ML engineering.
