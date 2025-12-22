# ClarityAI - Image Deblurring Application

A modern web application for AI-powered image deblurring using React and Azure Functions with deep learning.

🌐 **Live Demo**: https://black-forest-0e6a17503.3.azurestaticapps.net  
🚀 **Local Development**: `cd client && npm run dev` (see Quick Start below)

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
- **Azure Functions (Python)** - Serverless ML inference
- **Custom U-Net CNN** - Deep learning model for deblurring

### ML/AI
- **PyTorch** - Deep learning framework
- **U-Net Architecture** - 8.6M parameter CNN
- **GoPro_Large Dataset** - Trained on 3214 pairs of blurry-sharp images

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm (for React development)
- Python 3.10+ and Azure Functions Core Tools (optional, for local Azure Function development)

### Installation

```bash
# Install dependencies
npm install

# Start the React app
npm run dev
```

The app will be available at:
- **Frontend**: http://localhost:3000
- **Backend**: Azure Function (production deployment)

### Local Development with Azure Functions

To test with a local Azure Function:

```bash
# Terminal 1 - Start Azure Function locally (requires Azure Functions Core Tools)
cd function_app
func start

# Terminal 2 - Start React app and update the API endpoint in App.jsx to http://localhost:7071
cd client
npm run dev
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
├── function_app/        # Azure Function (Python) - Production Backend
│   ├── deblur_func/    # Deblurring endpoint
│   └── src/            # ML utilities (tiling, stitching)
│
└── src/                 # ML training code
    ├── train.py        # Training script
    ├── model_class.py  # U-Net architecture
    └── utils.py        # Data processing
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
- **Training**: GoPro_Large Dataset with 3214 pairs of blurry-sharp images
- **Performance**: PSNR 26.8 dB, SSIM 0.77
- **Inference Time**: 15-200 seconds (CPU on Azure Functions) depending on image size and cold start

## 🔧 API Endpoints

### Azure Function (Production Backend)
- `POST /api/imagedeblur` - Image deblurring endpoint
  - **Production URL**: https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/imagedeblur
  - **Local URL** (with Azure Functions Core Tools): http://localhost:7071/api/imagedeblur
- `GET /api/test` - Health check endpoint

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

MIT

