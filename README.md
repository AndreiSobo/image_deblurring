# CI 642 - Image Deblurring Application

A production-ready web application that restores clarity to blurry images using a custom-trained U-Net deep learning model. Built with React, Azure Functions, and PyTorch for real-world deployment.

🌐 **Live Demo**: https://black-forest-0e6a17503.3.azurestaticapps.net  
🚀 **Quick Start**: `npm run dev` (see below)

## ✨ Features

- 🎯 **Interactive Comparison Slider**: Drag to reveal dramatic before/after transformations
- 📱 **Fully Responsive**: Seamlessly adapts to mobile, tablet, and desktop
- ⚡ **Real-time Progress**: Live updates with processing time and status
- 🎨 **Modern UI/UX**: Clean, intuitive interface built with React + Tailwind CSS
- 📤 **Drag & Drop Upload**: Effortless image upload with smart file validation
- 🧠 **Production ML Pipeline**: Tiling & stitching for images of any size

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
- **PyTorch** - Deep learning framework with CPU/GPU support
- **Custom U-Net Architecture** - 8.6M parameter CNN with 4 encoder-decoder levels
- **GoPro_Large Dataset** - Trained on 3,214 image pairs (2,103 training + 1,111 validation)

## 🚀 Quick Start

### Prerequisites
- **Node.js 18+** and npm (for frontend development)
- **Python 3.10+** (optional, for ML model training or local function testing)

### Installation & Development

The project includes convenience scripts in the root `package.json` for easy setup:

```bash
# 1. Install all dependencies (root + client)
npm run install:all

# 2. Start development server
npm run dev
# → Opens at http://localhost:3000
# → Connects to production Azure Function backend
```

That's it! The app will hot-reload as you make changes.

### Additional Commands

```bash
# Build for production
npm run build

# Preview production build
npm run preview

# Or work directly in client folder
cd client && npm run dev
```

### Local Azure Function Development (Optional)

To test with a local backend (requires Azure Functions Core Tools):

```bash
cd function_app && func start
# Then update API endpoint in client/src/App.jsx to http://localhost:7071
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


### Responsive Design
Optimized layouts for all screen sizes with mobile-first approach.

## 🧠 ML Model Details

- **Architecture**: Custom U-Net with 4 encoder-decoder levels, skip connections, GroupNorm
- **Parameters**: 8.6M trainable parameters
- **Training Dataset**: GoPro_Large (2,103 training + 1,111 validation image pairs)
- **Loss Function**: Combined MS-SSIM (84%) + Charbonnier (16%)
- **Performance**: **PSNR 28.88 dB** | **SSIM 0.853** (best checkpoint at epoch 398)
- **Inference**: 12-250 seconds (Azure Functions CPU) depending on image size and cold start
- **Optimization**: Tiling & stitching for large images, gradient clipping, cosine annealing LR

> **Note**: Model trained with PyTorch 2.9.0 (CUDA) on RTX 4070, deployed with PyTorch 2.4.1 (CPU) for Azure compatibility.

## 🔧 API Endpoints

### Production Backend (Azure Functions)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/imagedeblur` | POST | Process and deblur images (accepts base64 JSON) |
| `/api/test` | GET | Health check and status endpoint |

**Base URL**: `https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net`  
**Local Development**: `http://localhost:7071` (when running Azure Functions locally)

## 📚 Documentation

For comprehensive technical details, see **[documentation/doc.md](documentation/doc.md)**:

- 🏗️ **Model Architecture**: U-Net design, layer configurations, normalization strategies
- 🎓 **Training Pipeline**: Dataset processing, augmentation, hyperparameter tuning
- 📊 **Loss Functions**: Combined MS-SSIM + Charbonnier loss evolution
- 🧩 **Tiling & Stitching**: Handling large images with overlapping tiles
- ☁️ **Azure Deployment**: Complete CI/CD pipeline, CORS setup, troubleshooting
- 📈 **MLflow Tracking**: Experiment management and model versioning
- 🔍 **Training Results**: 20+ experiments, convergence analysis, lessons learned

## 🌐 Resources

- **GoPro Dataset**: https://seungjunnah.github.io/Datasets/gopro.html

## 📄 License

MIT

