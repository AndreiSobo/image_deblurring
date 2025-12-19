# ClarityAI - Image Deblurring Application

A modern, full-stack web application for AI-powered image deblurring using React, Node.js, and Azure Functions with deep learning.

🌐 **Live Demo**: https://black-forest-0e6a17503.3.azurestaticapps.net (legacy static site)  
🚀 **React App**: Run locally with `npm run dev` (see Quick Start below)

## ✨ Features

- **Interactive Comparison Slider**: Drag to compare before/after images
- **Retro Terminal Animations**: 3 beautiful terminal themes during processing
- **Responsive Design**: Works seamlessly on mobile, tablet, and desktop
- **Real-time Processing**: Live progress updates with processing time
- **Modern UI/UX**: Built with React and Tailwind CSS
- **Drag & Drop Upload**: Easy image upload with file validation

## 🏗️ Technology Stack

### Frontend
- **React 18** - Modern UI library
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Beautiful icons

### Backend
- **Node.js + Express** - API server
- **Azure Functions (Python)** - Serverless ML inference
- **Custom U-Net CNN** - Deep learning model for deblurring

### ML/AI
- **PyTorch** - Deep learning framework
- **U-Net Architecture** - 8.6M parameter CNN
- **REDS Dataset** - Trained on 240 video sequences

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+ (for Azure Function local development)

### Installation

```bash
# Install all dependencies (root, client, and server)
npm run install:all

# Start both frontend and backend
npm run dev
```

The app will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

### Development

Run frontend and backend separately:

```bash
# Terminal 1 - Start backend
npm run dev:server

# Terminal 2 - Start frontend  
npm run dev:client
```

### Production Build

```bash
# Build React app for production
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
image_deblurring/
├── client/              # React frontend (Vite + Tailwind)
│   ├── src/
│   │   ├── App.jsx     # Main React component
│   │   ├── main.jsx    # Entry point
│   │   └── index.css   # Global styles
│   └── package.json
│
├── server/              # Node.js Express backend
│   ├── index.js        # API server with Azure Function proxy
│   └── package.json
│
├── function_app/        # Azure Function (Python)
│   ├── deblur_func/    # Deblurring endpoint
│   └── src/            # ML utilities (tiling, stitching)
│
├── src/                 # ML training code
│   ├── train.py        # Training script
│   ├── model_class.py  # U-Net architecture
│   └── utils.py        # Data processing
│
└── static/             # Legacy static site (deprecated)
```

## 🎨 Features Showcase

### Before/After Comparison Slider
Interactive slider to compare original and enhanced images with smooth drag functionality.

### Terminal Animation Themes
- **Classic** - Green terminal with scanlines
- **Amber** - Retro amber CRT display
- **Modern** - Clean gradient progress

### Responsive Design
Optimized layouts for all screen sizes with mobile-first approach.

## 🧠 ML Model Details

- **Architecture**: Custom U-Net with 4 encoder-decoder levels
- **Parameters**: 8.6M
- **Training**: REDS dataset (240 sequences)
- **Performance**: PSNR 26.8 dB, SSIM 0.77
- **Inference Time**: 15-25 seconds (CPU on Azure Functions)

## 🔧 API Endpoints

### Backend (Express - Port 5000)
- `GET /api/health` - Health check
- `POST /api/deblur` - Image deblurring endpoint

### Azure Function
- `POST /api/imagedeblur` - Direct deblurring endpoint
  - URL: https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/imagedeblur

## 📚 Documentation

For detailed technical documentation including:
- Model architecture and training
- Tiling & stitching algorithms
- Azure deployment guide
- Troubleshooting

See [documentation/doc.md](documentation/doc.md)

## 🌐 Resources

- **Legacy Static Site**: https://black-forest-0e6a17503.3.azurestaticapps.net
- **GoPro Dataset**: https://seungjunnah.github.io/Datasets/gopro.html

## 📄 License

This project is part of a portfolio demonstration and is available under the MIT License.

---

**Built with** ❤️ **using React, Node.js, PyTorch, and Azure**
