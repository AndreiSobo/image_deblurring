import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    Upload,
    Image as ImageIcon,
    Download,
    RefreshCw,
    Sparkles,
    AlertCircle,
    CheckCircle,
    Brain,
    Cpu,
    Target,
    ChevronDown,
    ChevronUp,
    X,
    MoveHorizontal,
    Terminal,
    Activity,
    Monitor,
    Settings
} from 'lucide-react';

// --- Components ---

const InfoCard = ({ icon: Icon, title, id, isOpen, toggle, children }) => (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 overflow-hidden transition-all duration-300 hover:shadow-md">
        <button
            onClick={() => toggle(id)}
            className="w-full flex items-center justify-between p-4 bg-slate-50/50 hover:bg-slate-50 transition-colors text-left"
        >
            <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 text-blue-600 rounded-lg">
                    <Icon size={20} />
                </div>
                <span className="font-semibold text-slate-700">{title}</span>
            </div>
            {isOpen ? <ChevronUp size={20} className="text-slate-400" /> : <ChevronDown size={20} className="text-slate-400" />}
        </button>
        <div
            className={`transition-all duration-300 ease-in-out overflow-hidden ${isOpen ? 'max-h-[800px] opacity-100' : 'max-h-0 opacity-0'
                }`}
        >
            <div className="p-5 text-slate-600 text-sm leading-relaxed border-t border-slate-100">
                {children}
            </div>
        </div>
    </div>
);

// Standardized container for all main views (Preview, Terminal, Result)
const DisplayStage = ({ children, className = "" }) => (
    <div className={`relative w-full h-[400px] md:h-[500px] mb-8 rounded-xl overflow-hidden shadow-2xl border border-slate-200 bg-slate-900 flex items-center justify-center ${className}`}>
        {children}
    </div>
);

const ComparisonSlider = ({ beforeImage, afterImage }) => {
    const [sliderPosition, setSliderPosition] = useState(50);
    const [containerWidth, setContainerWidth] = useState(0);
    const containerRef = useRef(null);
    const isDragging = useRef(false);

    const handleMove = useCallback((clientX) => {
        if (containerRef.current) {
            const rect = containerRef.current.getBoundingClientRect();
            const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
            const percentage = (x / rect.width) * 100;
            setSliderPosition(percentage);
        }
    }, []);

    const onMouseDown = () => (isDragging.current = true);
    const onMouseUp = () => (isDragging.current = false);
    const onMouseMove = (e) => {
        if (isDragging.current) handleMove(e.clientX);
    };

    // Touch support with preventDefault to avoid scrolling conflicts
    const onTouchStart = (e) => {
        isDragging.current = true;
        handleMove(e.touches[0].clientX);
    };

    const onTouchEnd = () => {
        isDragging.current = false;
    };

    const onTouchMove = (e) => {
        if (isDragging.current) {
            e.preventDefault(); // Prevent page scrolling while dragging
            handleMove(e.touches[0].clientX);
        }
    };

    // Update container width when component mounts or window resizes
    useEffect(() => {
        const updateWidth = () => {
            if (containerRef.current) {
                setContainerWidth(containerRef.current.offsetWidth);
            }
        };

        updateWidth();
        window.addEventListener('resize', updateWidth);
        document.addEventListener('mouseup', onMouseUp);
        document.addEventListener('touchend', onTouchEnd);

        return () => {
            window.removeEventListener('resize', updateWidth);
            document.removeEventListener('mouseup', onMouseUp);
            document.removeEventListener('touchend', onTouchEnd);
        };
    }, []);

    return (
        <div
            ref={containerRef}
            className="relative w-full h-full cursor-ew-resize select-none group"
            onMouseMove={onMouseMove}
            onMouseDown={onMouseDown}
            onTouchStart={onTouchStart}
            onTouchMove={onTouchMove}
            onTouchEnd={onTouchEnd}
        >
            {/* Before Image (Background) */}
            <img
                src={beforeImage}
                alt="Before"
                className="absolute top-0 left-0 w-full h-full object-contain bg-slate-900/50"
            />
            <div className="absolute top-4 left-4 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm z-20">Original</div>

            {/* After Image (Clipped on top) */}
            <div
                className="absolute top-0 left-0 h-full overflow-hidden"
                style={{ width: `${sliderPosition}%` }}
            >
                <img
                    src={afterImage}
                    alt="After"
                    className="h-full object-contain"
                    style={{
                        width: containerWidth ? `${containerWidth}px` : '100%',
                        maxWidth: 'none'
                    }}
                />
                <div className="absolute top-4 right-4 bg-blue-600/90 text-white text-xs px-2 py-1 rounded backdrop-blur-sm shadow-sm z-20">Enhanced</div>
            </div>

            {/* Slider Handle */}
            <div
                className="absolute top-0 bottom-0 w-1 bg-white/80 cursor-ew-resize shadow-[0_0_15px_rgba(0,0,0,0.5)] z-10 hover:bg-white transition-colors"
                style={{ left: `${sliderPosition}%` }}
            >
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center text-slate-800 transform group-hover:scale-110 transition-transform">
                    <MoveHorizontal size={16} />
                </div>
            </div>
        </div>
    );
};

// --- TERMINAL COMPONENT ---

const ProcessingTerminal = ({ progress }) => {
    const steps = [
        { threshold: 0, text: "initializing secure handshake..." },
        { threshold: 10, text: "sending image to azure function..." },
        { threshold: 20, text: "allocating tensor memory..." },
        { threshold: 25, text: "loading the CNN (U-Net)..." },
        { threshold: 35, text: "deblurring input..." },
        { threshold: 75, text: "optimizing pixels..." },
        { threshold: 80, text: "sending result to user..." }
    ];

    const activeSteps = steps.filter(step => progress >= step.threshold);

    return (
        <div className="w-full h-full flex flex-col bg-[#1e1e2e] font-sans tracking-wide relative">

            {/* Terminal Header */}
            <div className="p-3 flex items-center justify-between bg-[#2a2a3c] border-b border-white/5 shrink-0 z-10">
                <div className="flex gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500/80 hover:bg-red-500 transition-colors"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-500/80 hover:bg-yellow-500 transition-colors"></div>
                    <div className="w-3 h-3 rounded-full bg-green-500/80 hover:bg-green-500 transition-colors"></div>
                </div>

                <div className="text-[10px] uppercase flex items-center gap-2 text-slate-200 opacity-70">
                    <Activity size={10} />
                    <span>Processing Environment</span>
                </div>

                <div className="w-16"></div>
            </div>

            {/* Terminal Content */}
            <div className="flex-1 p-6 overflow-y-auto custom-scrollbar relative z-10">
                <div className="max-w-3xl mx-auto space-y-3">
                    {activeSteps.map((step, index) => (
                        <div key={index} className="flex items-start animate-in fade-in slide-in-from-left-2 duration-300">
                            <span className="mr-3 opacity-75 shrink-0 text-blue-400">➜</span>
                            <span className="text-slate-200">
                                {step.text}
                            </span>
                            {index === activeSteps.length - 1 && progress < 100 && (
                                <span className="ml-2 inline-block w-2 h-4 bg-gradient-to-r from-blue-500 to-violet-500 animate-pulse align-middle"></span>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Footer / Progress */}
            <div className="p-4 bg-[#2a2a3c] border-b border-white/5 border-t border-white/5 shrink-0 z-10">
                <div className="max-w-3xl mx-auto">
                    <div className="flex justify-between text-xs mb-2 opacity-70">
                        <span className="text-slate-200">STATUS: RUNNING</span>
                        <span className="text-slate-200">{Math.min(Math.round(progress), 100)}%</span>
                    </div>
                    <div className="h-1.5 w-full rounded-full overflow-hidden bg-blue-900/30">
                        <div
                            className="h-full transition-all duration-300 bg-gradient-to-r from-blue-500 to-violet-500"
                            style={{ width: `${Math.min(progress, 100)}%` }}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};

// --- Helper Functions ---

const compressImage = (file, maxWidth = 1920, maxHeight = 1920, quality = 0.85) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (event) => {
            const img = new Image();
            img.src = event.target.result;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;

                // Calculate new dimensions
                if (width > height) {
                    if (width > maxWidth) {
                        height = Math.round((height * maxWidth) / width);
                        width = maxWidth;
                    }
                } else {
                    if (height > maxHeight) {
                        width = Math.round((width * maxHeight) / height);
                        height = maxHeight;
                    }
                }

                canvas.width = width;
                canvas.height = height;

                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                canvas.toBlob(
                    (blob) => {
                        if (blob) {
                            const compressedFile = new File([blob], file.name, {
                                type: 'image/jpeg',
                                lastModified: Date.now(),
                            });
                            resolve(compressedFile);
                        } else {
                            reject(new Error('Canvas to Blob conversion failed'));
                        }
                    },
                    'image/jpeg',
                    quality
                );
            };
            img.onerror = reject;
        };
        reader.onerror = reject;
    });
};

// --- Main App Component ---

export default function ImageDeblurApp() {
    const [file, setFile] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);
    const [processedImage, setProcessedImage] = useState(null);
    const [status, setStatus] = useState('idle'); // idle, processing, success, error
    const [progress, setProgress] = useState(0);
    const [metrics, setMetrics] = useState(null);
    const [activeInfoCard, setActiveInfoCard] = useState('what');
    const [errorMessage, setErrorMessage] = useState('');

    const fileInputRef = useRef(null);

    const toggleInfoCard = (id) => {
        setActiveInfoCard(activeInfoCard === id ? null : id);
    };

    const handleFileSelect = async (selectedFile) => {
        if (!selectedFile) return;

        if (!selectedFile.type.startsWith('image/')) {
            alert('Please upload a valid image file');
            return;
        }

        if (selectedFile.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB');
            return;
        }

        // Compress image if it's too large (especially for mobile photos)
        let processedFile = selectedFile;
        if (selectedFile.size > 2 * 1024 * 1024) { // If larger than 2MB
            try {
                processedFile = await compressImage(selectedFile);
            } catch (err) {
                console.warn('Image compression failed, using original:', err);
            }
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            setFile(processedFile);
            setImagePreview(e.target.result);
            setProcessedImage(null);
            setStatus('idle');
            setProgress(0);
            setErrorMessage('');
        };
        reader.readAsDataURL(processedFile);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    };

    const handleProcess = async () => {
        setStatus('processing');
        setProgress(0);
        setErrorMessage('');

        // Progress simulation
        const interval = setInterval(() => {
            setProgress((prev) => {
                if (prev >= 98) return prev;
                const increment = prev > 40 && prev < 80 ? Math.random() * 2 : Math.random() * 8;
                return prev + increment;
            });
        }, 400);

        try {
            const startTime = Date.now();

            // Convert image to base64
            const reader = new FileReader();
            const base64Promise = new Promise((resolve, reject) => {
                reader.onload = () => {
                    const base64 = reader.result.split(',')[1]; // Remove data:image/...;base64, prefix
                    resolve(base64);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });

            const base64Image = await base64Promise;

            // Log request details for debugging
            console.log('Sending image to API:', {
                imageSize: `${(base64Image.length / 1024).toFixed(2)} KB`,
                fileType: file.type,
                fileName: file.name,
                userAgent: navigator.userAgent.substring(0, 50)
            });

            // Use relative URL - will work both in dev (via proxy) and production (Azure Static Web App routing)
            // This ensures consistent behavior across desktop and mobile
            const apiUrl = '/api/imagedeblur';

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: base64Image }),
            });

            if (!response.ok) {
                let errorMessage = `Server error: ${response.status}`;
                try {
                    const error = await response.json();
                    errorMessage = error.error || error.message || errorMessage;
                } catch (jsonError) {
                    // If response is not JSON (HTML error page, empty response, etc.)
                    const text = await response.text();
                    console.error('Non-JSON error response:', text.substring(0, 200));
                    errorMessage = `Server error (${response.status}): ${text.substring(0, 100)}`;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            const processingTime = ((Date.now() - startTime) / 1000).toFixed(1);

            clearInterval(interval);

            // Set all states first, then set progress to 100% last
            // This ensures progress reaches 100% only when everything is ready to display
            setProcessedImage(data.deblurred_image);
            setMetrics({
                time: `${processingTime}s`,
                quality: data.quality || '94%'
            });
            setStatus('success');

            // Small delay to ensure state updates are processed before showing 100%
            setTimeout(() => {
                setProgress(100);
            }, 100);

        } catch (error) {
            console.error('Error processing image:', error);
            clearInterval(interval);
            setStatus('error');

            // More descriptive error messages for mobile debugging
            let userMessage = 'Something went wrong. Please try again.';
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                userMessage = 'Network error. Please check your internet connection and try again.';
            } else if (error.message.includes('timeout')) {
                userMessage = 'Request timed out. Please try a smaller image.';
            } else if (error.message) {
                userMessage = error.message;
            }

            setErrorMessage(userMessage);
        }
    };

    const resetApp = () => {
        setFile(null);
        setImagePreview(null);
        setProcessedImage(null);
        setStatus('idle');
        setMetrics(null);
        setErrorMessage('');
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 text-slate-800 font-sans selection:bg-blue-100 pb-12">

            {/* Header */}
            <header className="bg-white/80 backdrop-blur-md sticky top-0 z-50 border-b border-slate-200 shadow-sm">
                <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="bg-blue-600 text-white p-2 rounded-lg">
                            <Sparkles size={24} />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-indigo-600">
                                CI 642
                            </h1>
                            <p className="text-xs text-slate-500 font-medium">Deep Learning Image Deblurring</p>
                        </div>
                    </div>
                    <a href="https://github.com" className="text-sm font-medium text-slate-500 hover:text-blue-600 transition-colors">
                        View on GitHub
                    </a>
                </div>
            </header>

            <main className="max-w-6xl mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* Left Column: Info & Context */}
                <div className="lg:col-span-4 space-y-4">
                    <div className="lg:sticky lg:top-28 space-y-4">
                        <InfoCard
                            icon={Target}
                            title="What is this?"
                            id="what"
                            isOpen={activeInfoCard === 'what'}
                            toggle={toggleInfoCard}
                        >
                            <p>A webapp that uses a <strong>custom U-Net neural network</strong> to restore sharpness to blurred images. It tiles large images and processes the tiles via serverless functions.</p>
                            <ul className="mt-3 space-y-1 list-disc pl-4 marker:text-blue-500">
                                <li>Removes motion & defocus blur</li>
                                <li>Supports images of different sizes</li>
                                <li>Secure, serverless processing</li>
                                <li>Inference time 15 - 150 seconds</li>
                            </ul>
                        </InfoCard>

                        <InfoCard
                            icon={Cpu}
                            title="How it works"
                            id="how"
                            isOpen={activeInfoCard === 'how'}
                            toggle={toggleInfoCard}
                        >
                            <p className="bg-slate-100 p-2 rounded text-xs font-mono mb-3 text-blue-700">
                                Upload → Base64 → Azure Function → U-Net Inference → Result
                            </p>
                            <p>The project uses a PyTorch model trained on the GoPro_Large dataset. The architecture features a 32-channel Convolutional Neural Network optimized for CPU inference on standard cloud tiers.</p>
                        </InfoCard>

                        <InfoCard
                            icon={Brain}
                            title="Why it matters"
                            id="why"
                            isOpen={activeInfoCard === 'why'}
                            toggle={toggleInfoCard}
                        >
                            <p>This project demonstrates end-to-end ML engineering: from training custom models to deploying scalable cloud infrastructure and building responsive frontends.</p>
                        </InfoCard>
                    </div>
                </div>

                {/* Right Column: Application */}
                <div className="lg:col-span-8">
                    <div className="bg-white rounded-2xl shadow-xl shadow-blue-900/5 border border-white overflow-hidden">

                        {/* Header of App Area */}
                        <div className="p-6 border-b border-slate-100 flex justify-between items-center">
                            <h2 className="text-lg font-bold text-slate-800">Image Workspace</h2>
                            {file && (
                                <button
                                    onClick={resetApp}
                                    className="text-xs flex items-center gap-1 text-slate-400 hover:text-red-500 transition-colors"
                                >
                                    <X size={14} /> Clear
                                </button>
                            )}
                        </div>

                        <div className="p-6 md:p-8 flex flex-col items-center">

                            {/* STATE: IDLE (Upload) */}
                            {!file && (
                                <div
                                    className="w-full h-[400px] md:h-[500px] border-2 border-dashed border-slate-200 rounded-2xl flex flex-col items-center justify-center bg-slate-50/50 hover:bg-blue-50/30 hover:border-blue-300 transition-all cursor-pointer group mb-8"
                                    onDragOver={(e) => e.preventDefault()}
                                    onDrop={handleDrop}
                                    onClick={() => fileInputRef.current.click()}
                                >
                                    <div className="w-20 h-20 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-sm">
                                        <Upload size={32} />
                                    </div>
                                    <h3 className="text-xl font-bold text-slate-700 mb-2">Upload your blurred image</h3>
                                    <p className="text-slate-500 mb-6 text-center max-w-sm">
                                        Drag & drop or click to browse. <br />
                                        <span className="text-xs">Supports JPG, PNG, WebP up to 10MB</span>
                                    </p>
                                    <button className="px-6 py-2 bg-white border border-slate-200 text-slate-700 font-medium rounded-lg shadow-sm group-hover:shadow-md transition-all">
                                        Browse Files
                                    </button>
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        className="hidden"
                                        accept="image/*"
                                        onChange={(e) => handleFileSelect(e.target.files[0])}
                                    />
                                </div>
                            )}

                            {/* STATE: PREVIEW */}
                            {file && !processedImage && status === 'idle' && (
                                <>
                                    <DisplayStage>
                                        <img src={imagePreview} alt="Preview" className="w-full h-full object-contain" />
                                        <div className="absolute bottom-0 left-0 right-0 bg-black/60 backdrop-blur-sm text-white p-2 text-xs flex justify-between px-4 z-10">
                                            <span>{file.name}</span>
                                            <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                                        </div>
                                    </DisplayStage>
                                    <button
                                        onClick={handleProcess}
                                        className="w-full max-w-xs py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-lg shadow-blue-500/30 transition-all transform hover:-translate-y-0.5 flex items-center justify-center gap-2"
                                    >
                                        <Sparkles size={20} /> Deblur Image
                                    </button>
                                </>
                            )}

                            {/* STATE: PROCESSING (Terminal) */}
                            {status === 'processing' && (
                                <DisplayStage>
                                    <ProcessingTerminal progress={progress} />
                                </DisplayStage>
                            )}

                            {/* STATE: ERROR */}
                            {status === 'error' && (
                                <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-xl flex items-center gap-3 w-full">
                                    <AlertCircle size={20} />
                                    <span>{errorMessage}</span>
                                </div>
                            )}

                            {/* STATE: SUCCESS (Result) */}
                            {processedImage && (
                                <div className="w-full animate-in fade-in slide-in-from-bottom-4 duration-700">
                                    <div className="flex flex-wrap items-center justify-between mb-4 gap-4">
                                        <div className="flex items-center gap-2 text-green-600 bg-green-50 px-3 py-1 rounded-full text-sm font-medium">
                                            <CheckCircle size={16} />
                                            Processing Complete
                                        </div>
                                        {metrics && metrics.time && (
                                            <div className="text-xs text-slate-400 font-mono">
                                                Processing time: {metrics.time}
                                            </div>
                                        )}
                                    </div>

                                    {/* Comparison Component - Wrapped in DisplayStage for consistent sizing */}
                                    <DisplayStage>
                                        <ComparisonSlider beforeImage={imagePreview} afterImage={processedImage} />
                                    </DisplayStage>

                                    <div className="text-center text-xs text-slate-400 mt-2 mb-6 flex items-center justify-center gap-2">
                                        <MoveHorizontal size={14} /> Drag slider to compare
                                    </div>

                                    <div className="flex gap-4 justify-center">
                                        <a
                                            href={processedImage}
                                            download={`enhanced_${file.name}`}
                                            className="flex-1 max-w-xs py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-lg shadow-blue-500/30 transition-all flex items-center justify-center gap-2"
                                        >
                                            <Download size={20} /> Download
                                        </a>
                                        <button
                                            onClick={resetApp}
                                            className="px-4 py-3 bg-white border border-slate-200 text-slate-700 font-medium rounded-xl hover:bg-slate-50 transition-all"
                                        >
                                            <RefreshCw size={20} />
                                        </button>
                                    </div>
                                </div>
                            )}

                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
