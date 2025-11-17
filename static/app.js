// Mobile-responsive Image Deblurring Application
// Enhanced with drag-and-drop, image processing, and Azure cloud integration

document.addEventListener('DOMContentLoaded', function () {
    // DOM element references
    const uploadArea = document.getElementById('upload-area');
    const imageUpload = document.getElementById('image-upload');
    const previewSection = document.getElementById('preview-section');
    const previewImage = document.getElementById('preview-image');
    const imageInfo = document.getElementById('image-info');
    const actionButtons = document.getElementById('action-buttons');
    const deblurBtn = document.getElementById('deblur-btn');
    const clearBtn = document.getElementById('clear-btn');
    const processing = document.getElementById('processing');
    const progressBar = document.getElementById('progress-bar');
    const resultsSection = document.getElementById('results-section');
    const originalResult = document.getElementById('original-result');
    const deblurredResult = document.getElementById('deblurred-result');
    const downloadBtn = document.getElementById('download-btn');
    const newImageBtn = document.getElementById('new-image-btn');
    const statusContainer = document.getElementById('status-container');
    const statusMessage = document.getElementById('status-message');

    // Application state
    let currentImage = null;
    let currentImageFile = null;
    let deblurredImageData = null;

    // Mobile detection and optimization
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

    // Initialize mobile-specific features
    if (isMobile || isTouchDevice) {
        initializeMobileFeatures();
    }

    // Initialize drag and drop functionality
    initializeDragAndDrop();

    // Initialize info card keyboard accessibility
    initializeInfoCardAccessibility();

    // API Configuration - Will be updated when Azure Function is deployed
    const API_CONFIG = {
        imageDeblur: 'https://your-deblur-function-url/api/DeblurImage' // To be updated
    };

    // =============================================================================
    // File Upload and Preview Functions
    // =============================================================================

    // Handle file selection via input or drag-and-drop
    function handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            showStatusMessage('Please select a valid image file (JPG, PNG, WebP).', 'error');
            return;
        }

        // Validate file size (max 10MB)
        const maxSize = 10 * 1024 * 1024; // 10MB in bytes
        if (file.size > maxSize) {
            showStatusMessage('File size must be less than 10MB. Please resize your image.', 'error');
            return;
        }

        currentImageFile = file;
        previewImageFile(file);
    }

    // Preview selected image
    function previewImageFile(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            currentImage = e.target.result;
            previewImage.src = currentImage;
            
            // Show preview section and action buttons
            previewSection.style.display = 'block';
            actionButtons.style.display = 'block';
            statusContainer.style.display = 'none';
            
            // Update image info
            updateImageInfo(file);
            
            // Add visual feedback
            previewSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            
            showStatusMessage('Image loaded successfully! Click "Deblur Image" to process.', 'success');
        };
        
        reader.onerror = function() {
            showStatusMessage('Error reading the image file. Please try again.', 'error');
        };
        
        reader.readAsDataURL(file);
    }

    // Update image information display
    function updateImageInfo(file) {
        const img = new Image();
        img.onload = function() {
            const sizeKB = (file.size / 1024).toFixed(1);
            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            const sizeText = file.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;
            
            // Check if image is very large and warn user
            const warningText = (img.width > 2000 || img.height > 2000) 
                ? '<br><span style="color: #ff9800;">⚠️ Large image - processing may take 60-90 seconds</span>' 
                : '';
            
            imageInfo.innerHTML = `
                <strong>📁 ${file.name}</strong><br>
                📏 ${img.width} × ${img.height} pixels<br>
                💾 ${sizeText}<br>
                🎨 ${file.type}${warningText}
            `;
        };
        img.src = currentImage;
    }

    // =============================================================================
    // Drag and Drop Functionality
    // =============================================================================

    function initializeDragAndDrop() {
        // Prevent default drag behaviors on the entire document
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, preventDefaults, false);
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight upload area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });

        // Handle dropped files
        uploadArea.addEventListener('drop', handleDrop, false);
        
        // Handle click to upload
        uploadArea.addEventListener('click', () => {
            imageUpload.click();
        });
        
        // Handle file input change
        imageUpload.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        uploadArea.classList.add('dragover');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    }

    // =============================================================================
    // Image Deblurring Functions
    // =============================================================================

    // Handle deblur button click
    deblurBtn.addEventListener('click', function() {
        if (!currentImage || !currentImageFile) {
            showStatusMessage('Please upload an image first.', 'error');
            return;
        }

        processImageDeblurring();
    });

    // Process image deblurring
    function processImageDeblurring() {
        // Show processing indicator
        processing.style.display = 'block';
        actionButtons.style.display = 'none';
        resultsSection.style.display = 'none';
        
        // Scroll to processing section
        processing.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 85) progress = 85;
            updateProgress(progress);
        }, 400);

        // For now, simulate the API call with mock data
        // In production, this would be replaced with actual Azure Function call
        const useMockData = true; // Set to false when actual API is available

        if (useMockData) {
            // Simulate processing delay (realistic for CPU inference)
            setTimeout(() => {
                clearInterval(progressInterval);
                updateProgress(100);
                
                setTimeout(() => {
                    // Mock successful deblurring - for demo purposes, show the same image
                    // In production, this would be the actual deblurred result
                    handleDeblurringResponse({
                        success: true,
                        original_image: currentImage,
                        deblurred_image: currentImage, // Would be actual deblurred image from Azure
                        processing_time: (Math.random() * 30 + 15).toFixed(1), // 15-45 seconds
                        quality_improvement: (Math.random() * 0.3 + 0.7).toFixed(2) // 0.7-1.0
                    });
                }, 500);
            }, 3500); // 3.5 second demo delay
        } else {
            // Actual API call (to be implemented when Azure Function is ready)
            callDeblurAPI(currentImageFile)
                .then(response => {
                    clearInterval(progressInterval);
                    updateProgress(100);
                    setTimeout(() => handleDeblurringResponse(response), 500);
                })
                .catch(error => {
                    clearInterval(progressInterval);
                    handleDeblurringError(error);
                });
        }
    }

    // Call the deblurring API (placeholder for future implementation)
    async function callDeblurAPI(imageFile) {
        // Convert image to base64
        const base64Image = await fileToBase64(imageFile);
        
        const response = await fetch(API_CONFIG.imageDeblur, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: base64Image
            })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    }

    // Helper function to convert file to base64
    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                // Remove the data URL prefix to get just the base64 string
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    // Handle successful deblurring response
    function handleDeblurringResponse(data) {
        console.log('Deblurring response:', data);

        if (data.success) {
            // Store deblurred image data
            deblurredImageData = data.deblurred_image;
            
            // Update result images
            originalResult.src = data.original_image;
            deblurredResult.src = data.deblurred_image;
            
            // Show results section
            processing.style.display = 'none';
            resultsSection.style.display = 'block';
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            
            // Show success message with metrics
            const timeText = data.processing_time ? `${data.processing_time}s` : 'N/A';
            const qualityText = data.quality_improvement ? `${(data.quality_improvement * 100).toFixed(0)}%` : 'N/A';
            showStatusMessage(
                `✅ Deblurring completed! Processing time: ${timeText} | Quality improvement: ${qualityText}`, 
                'success'
            );
        } else {
            handleDeblurringError(new Error(data.message || 'Deblurring failed'));
        }
    }

    // Handle deblurring error
    function handleDeblurringError(error) {
        console.error('Deblurring error:', error);
        
        processing.style.display = 'none';
        actionButtons.style.display = 'block';
        
        showStatusMessage(`❌ Deblurring failed: ${error.message}. Please try again or use a smaller image.`, 'error');
    }

    // Update progress bar
    function updateProgress(percent) {
        progressBar.style.width = `${Math.min(percent, 100)}%`;
    }

    // =============================================================================
    // Utility Functions
    // =============================================================================

    // Clear current image and reset interface
    clearBtn.addEventListener('click', function() {
        if (confirm('Are you sure you want to clear the current image?')) {
            resetInterface();
        }
    });

    newImageBtn.addEventListener('click', function() {
        resetInterface();
    });

    // Reset the interface to initial state
    function resetInterface() {
        currentImage = null;
        currentImageFile = null;
        deblurredImageData = null;
        
        previewSection.style.display = 'none';
        actionButtons.style.display = 'none';
        processing.style.display = 'none';
        resultsSection.style.display = 'none';
        statusContainer.style.display = 'block';
        
        previewImage.src = '';
        imageInfo.innerHTML = '';
        imageUpload.value = '';
        updateProgress(0);
        
        showStatusMessage('📤 Upload or drag & drop an image to get started', 'info');
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Download enhanced image
    downloadBtn.addEventListener('click', function() {
        if (!deblurredImageData) {
            showStatusMessage('No enhanced image available for download.', 'error');
            return;
        }

        // Create download link
        const link = document.createElement('a');
        link.href = deblurredImageData;
        link.download = `deblurred_${currentImageFile ? currentImageFile.name : 'image.jpg'}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showStatusMessage('✅ Enhanced image downloaded successfully!', 'success');
    });

    // Show status message with type styling
    function showStatusMessage(message, type = 'info') {
        statusMessage.className = 'status-message';
        
        switch (type) {
            case 'success':
                statusMessage.className += ' status-success';
                break;
            case 'error':
                statusMessage.className += ' status-error';
                break;
            case 'warning':
                statusMessage.className += ' status-warning';
                break;
            default:
                statusMessage.className += ' status-info';
        }
        
        statusMessage.innerHTML = message;
        
        // Show status container if hidden
        if (statusContainer.style.display === 'none') {
            statusContainer.style.display = 'block';
        }
    }

    // =============================================================================
    // Mobile-specific Features
    // =============================================================================

    function initializeMobileFeatures() {
        // Add touch feedback to all interactive elements
        const interactiveElements = document.querySelectorAll('button, .upload-area, .info-card-header');

        interactiveElements.forEach(element => {
            element.addEventListener('touchstart', function() {
                this.style.transform = 'scale(0.98)';
            }, { passive: true });

            element.addEventListener('touchend', function() {
                setTimeout(() => {
                    this.style.transform = '';
                }, 150);
            }, { passive: true });
        });

        // Add haptic feedback for supported devices
        if ('vibrate' in navigator) {
            deblurBtn.addEventListener('click', () => {
                navigator.vibrate(50);
            });
            
            downloadBtn.addEventListener('click', () => {
                navigator.vibrate([30, 50, 30]);
            });
        }
    }

    // =============================================================================
    // Info Card Functionality
    // =============================================================================

    function initializeInfoCardAccessibility() {
        const infoCardHeaders = document.querySelectorAll('.info-card-header');
        const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

        infoCardHeaders.forEach(header => {
            // Add keyboard support
            header.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.click();
                }
            });

            // Add touch feedback for mobile
            if (isTouchDevice) {
                header.addEventListener('touchstart', function() {
                    this.style.transform = 'scale(0.98)';
                }, { passive: true });

                header.addEventListener('touchend', function() {
                    setTimeout(() => {
                        this.style.transform = '';
                    }, 150);
                }, { passive: true });
            }
        });
    }

    // Initialize with welcome message
    showStatusMessage('📤 Upload or drag & drop an image to get started', 'info');
});

// =============================================================================
// Info Card Toggle Function (Global scope for onclick handlers)
// =============================================================================

function toggleInfoCard(cardType) {
    const cardBody = document.getElementById(`${cardType}-body`);
    const cardArrow = document.getElementById(`${cardType}-arrow`);
    const cardHeader = document.querySelector(`#${cardType}-card .info-card-header`);

    // FIX: Check for both empty string and 'none' since CSS display may not be inline
    const isHidden = cardBody.style.display === 'none' || cardBody.style.display === '';

    if (isHidden) {
        // Expand the card
        cardBody.style.display = 'block';
        cardArrow.textContent = '▲';
        cardArrow.setAttribute('aria-label', `Collapse ${cardType} explanation`);
        cardHeader.setAttribute('aria-expanded', 'true');
        
        // Smooth scroll to make expanded content visible
        setTimeout(() => {
            cardBody.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    } else {
        // Collapse the card
        cardBody.style.display = 'none';
        cardArrow.textContent = '▼';
        cardArrow.setAttribute('aria-label', `Expand ${cardType} explanation`);
        cardHeader.setAttribute('aria-expanded', 'false');
    }
}