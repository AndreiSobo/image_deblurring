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

    // Touch support
    const onTouchMove = (e) => handleMove(e.touches[0].clientX);

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

        return () => {
            window.removeEventListener('resize', updateWidth);
            document.removeEventListener('mouseup', onMouseUp);
        };
    }, []);

    return (
        <div
            ref={containerRef}
            className="relative w-full h-full cursor-ew-resize select-none group"
            onMouseMove={onMouseMove}
            onMouseDown={onMouseDown}
            onTouchMove={onTouchMove}
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

// --- TERMINAL VARIANTS ---

const RetroTerminal = ({ progress, variant = 'classic', onVariantChange }) => {
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

    // Style configurations
    const styles = {
        classic: {
            bg: "bg-slate-900",
            header: "bg-slate-800 border-b border-slate-700",
            text: "text-green-400",
            accent: "text-blue-400",
            progressBg: "bg-green-500/20",
            progressBar: "bg-green-500",
            glow: "0 0 5px rgba(74, 222, 128, 0.5)",
            scanline: true,
            font: "font-mono"
        },
        amber: {
            bg: "bg-[#1a1200]", // Very dark amber/brown
            header: "bg-[#2e2000] border-b border-amber-900/50",
            text: "text-amber-500",
            accent: "text-amber-300",
            progressBg: "bg-amber-900/30",
            progressBar: "bg-amber-500",
            glow: "0 0 8px rgba(245, 158, 11, 0.6)",
            scanline: true,
            font: "font-mono tracking-wider"
        },
        modern: {
            bg: "bg-[#1e1e2e]", // Dark cool blue/purple
            header: "bg-[#2a2a3c] border-b border-white/5",
            text: "text-slate-200",
            accent: "text-blue-400",
            progressBg: "bg-blue-900/30",
            progressBar: "bg-gradient-to-r from-blue-500 to-violet-500",
            glow: "none",
            scanline: false,
            font: "font-sans tracking-wide"
        }
    };

    const currentStyle = styles[variant] || styles.classic;

    return (
        <div className={`w-full h-full flex flex-col ${currentStyle.bg} ${currentStyle.font} relative`}>

            {/* Terminal Header */}
            <div className={`p-3 flex items-center justify-between ${currentStyle.header} shrink-0 z-10`}>
                <div className="flex gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500/80 hover:bg-red-500 transition-colors"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-500/80 hover:bg-yellow-500 transition-colors"></div>
                    <div className="w-3 h-3 rounded-full bg-green-500/80 hover:bg-green-500 transition-colors"></div>
                </div>

                <div className={`text-[10px] uppercase flex items-center gap-2 ${currentStyle.text} opacity-70`}>
                    <Activity size={10} />
                    <span>Processing Environment</span>
                </div>

                {/* Variant Switcher */}
                <div className="flex gap-1">
                    {Object.keys(styles).map((v) => (
                        <button
                            key={v}
                            onClick={() => onVariantChange(v)}
                            className={`w-4 h-4 rounded border ${variant === v ? 'border-white bg-white/20' : 'border-white/10 hover:bg-white/10'} transition-all`}
                            title={`Switch to ${v} theme`}
                        />
                    ))}
                </div>
            </div>

            {/* Terminal Content */}
            <div className="flex-1 p-6 overflow-y-auto custom-scrollbar relative z-10">
                <div className="max-w-3xl mx-auto space-y-3">
                    {activeSteps.map((step, index) => (
                        <div key={index} className="flex items-start animate-in fade-in slide-in-from-left-2 duration-300">
                            <span className={`mr-3 opacity-75 shrink-0 ${currentStyle.accent}`}>➜</span>
                            <span
                                className={currentStyle.text}
                                style={{ textShadow: currentStyle.glow }}
                            >
                                {step.text}
                            </span>
                            {index === activeSteps.length - 1 && progress < 100 && (
                                <span className={`ml-2 inline-block w-2 h-4 ${currentStyle.progressBar} animate-pulse align-middle`}></span>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Footer / Progress */}
            <div className={`p-4 ${currentStyle.header} border-t border-white/5 shrink-0 z-10`}>
                <div className="max-w-3xl mx-auto">
                    <div className="flex justify-between text-xs mb-2 opacity-70">
                        <span className={currentStyle.text}>STATUS: RUNNING</span>
                        <span className={currentStyle.text}>{Math.round(progress)}%</span>
                    </div>
                    <div className={`h-1.5 w-full rounded-full overflow-hidden ${currentStyle.progressBg}`}>
                        <div
                            className={`h-full transition-all duration-300 ${currentStyle.progressBar}`}
                            style={{ width: `${progress}%` }}
                        >
                            {variant !== 'modern' && <div className="absolute inset-0 bg-white/20 w-full animate-shimmer"></div>}
                        </div>
                    </div>
                </div>
            </div>

            {/* Scanline Effect Overlay (Conditional) */}
            {currentStyle.scanline && (
                <div
                    className="absolute inset-0 pointer-events-none opacity-10 z-0"
                    style={{
                        background: 'linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))',
                        backgroundSize: '100% 2px, 3px 100%'
                    }}
                ></div>
            )}
        </div>
    );
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
    const [terminalVariant, setTerminalVariant] = useState('classic');
    const [errorMessage, setErrorMessage] = useState('');

    const fileInputRef = useRef(null);

    const toggleInfoCard = (id) => {
        setActiveInfoCard(activeInfoCard === id ? null : id);
    };

    const handleFileSelect = (selectedFile) => {
        if (!selectedFile) return;

        if (!selectedFile.type.startsWith('image/')) {
            alert('Please upload a valid image file');
            return;
        }

        if (selectedFile.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            setFile(selectedFile);
            setImagePreview(e.target.result);
            setProcessedImage(null);
            setStatus('idle');
            setProgress(0);
            setErrorMessage('');
        };
        reader.readAsDataURL(selectedFile);
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

            // Send directly to Azure Function
            const response = await fetch('https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/imagedeblur', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: base64Image }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Processing failed');
            }

            const data = await response.json();
            const processingTime = ((Date.now() - startTime) / 1000).toFixed(1);

            clearInterval(interval);
            setProgress(100);
            setProcessedImage(data.deblurred_image);
            setMetrics({
                time: `${processingTime}s`,
                quality: data.quality || '94%'
            });
            setStatus('success');

        } catch (error) {
            console.error('Error processing image:', error);
            clearInterval(interval);
            setStatus('error');
            setErrorMessage(error.message || 'Something went wrong. Please try a smaller image.');
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
                            <p>This tool uses a <strong>custom U-Net neural network</strong> to restore sharpness to blurred images. It intelligently tiles large images and processes them via serverless functions.</p>
                            <ul className="mt-3 space-y-1 list-disc pl-4 marker:text-blue-500">
                                <li>Removes motion & defocus blur</li>
                                <li>Supports 4K resolution</li>
                                <li>Secure, serverless processing</li>
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
                            <p>We use a PyTorch model trained on the REDS dataset. The architecture features a 32-channel U-Net with GroupNorm, optimized for CPU inference on standard cloud tiers.</p>
                        </InfoCard>

                        <InfoCard
                            icon={Brain}
                            title="Why it matters"
                            id="why"
                            isOpen={activeInfoCard === 'why'}
                            toggle={toggleInfoCard}
                        >
                            <p>This project demonstrates end-to-end ML engineering: from training custom architectures to deploying scalable cloud infrastructure and building responsive frontends.</p>
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
                                    <RetroTerminal
                                        progress={progress}
                                        variant={terminalVariant}
                                        onVariantChange={setTerminalVariant}
                                    />
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
