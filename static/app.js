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
        deblur: 'https://your-deblur-function-url/api/DeblurImage' // To be updated
    };

    // =============================================================================
    // File Upload and Preview Functions
    // =============================================================================

    // Handle file selection via input or drag-and-drop
    function handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            showStatusMessage('Please select a valid image file.', 'error');
            return;
        }

        // Validate file size (max 10MB)
        const maxSize = 10 * 1024 * 1024; // 10MB in bytes
        if (file.size > maxSize) {
            showStatusMessage('File size must be less than 10MB.', 'error');
            return;
        }

        currentImageFile = file;
        previewImage(file);
    }

    // Preview selected image
    function previewImage(file) {
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
            previewSection.scrollIntoView({ behavior: 'smooth' });
        };
        
        reader.onerror = function() {
            showStatusMessage('Error reading the image file.', 'error');
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
            
            imageInfo.innerHTML = `
                <strong>📁 ${file.name}</strong><br>
                📏 ${img.width} × ${img.height} pixels<br>
                💾 ${sizeText}<br>
                🎨 ${file.type}
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
        uploadArea.addEventListener('click', () => imageUpload.click());
        
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
        
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            updateProgress(progress);
        }, 300);

        // For now, simulate the API call with mock data
        // In production, this would be replaced with actual Azure Function call
        const useMockData = true; // Set to false when actual API is available

        if (useMockData) {
            // Simulate processing delay
            setTimeout(() => {
                clearInterval(progressInterval);
                updateProgress(100);
                
                setTimeout(() => {
                    // Mock successful deblurring - for demo purposes, show the same image
                    // In production, this would be the actual deblurred result
                    handleDeblurringResponse({
                        success: true,
                        original_image: currentImage,
                        deblurred_image: currentImage, // Would be actual deblurred image
                        processing_time: 2.3,
                        quality_improvement: 0.85
                    });
                }, 500);
            }, 3000);
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
        const formData = new FormData();
        formData.append('image', imageFile);

        const response = await fetch(API_CONFIG.deblur, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        return await response.json();
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
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            
            // Show success message
            showStatusMessage(`Deblurring completed successfully! Processing time: ${data.processing_time || 'N/A'}s`, 'success');
        } else {
            handleDeblurringError(new Error(data.message || 'Deblurring failed'));
        }
    }

    // Handle deblurring error
    function handleDeblurringError(error) {
        console.error('Deblurring error:', error);
        
        processing.style.display = 'none';
        actionButtons.style.display = 'block';
        
        showStatusMessage(`Deblurring failed: ${error.message}`, 'error');
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
        resetInterface();
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
        
        showStatusMessage('Upload an image to get started', 'info');
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
        link.download = `enhanced_${currentImageFile ? currentImageFile.name : 'image.jpg'}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showStatusMessage('Enhanced image downloaded successfully!', 'success');
    });

    // Show status message with type styling
    function showStatusMessage(message, type = 'info') {
        statusMessage.className = 'h4 text-center';
        
        switch (type) {
            case 'success':
                statusMessage.className += ' text-success';
                statusMessage.innerHTML = `✅ ${message}`;
                break;
            case 'error':
                statusMessage.className += ' text-danger';
                statusMessage.innerHTML = `❌ ${message}`;
                break;
            case 'warning':
                statusMessage.className += ' text-warning';
                statusMessage.innerHTML = `⚠️ ${message}`;
                break;
            default:
                statusMessage.className += ' text-muted';
                statusMessage.innerHTML = message;
        }
    }

    // =============================================================================
    // Mobile-specific Features
    // =============================================================================

    function initializeMobileFeatures() {
        // Add touch feedback to all interactive elements
        const interactiveElements = document.querySelectorAll('button, .upload-area');

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
        }
    }

    // =============================================================================
    // Info Card Functionality (Reused from web_app)
    // =============================================================================

    function initializeInfoCardAccessibility() {
        const infoCardHeaders = document.querySelectorAll('.info-card-header');
        const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

        infoCardHeaders.forEach(header => {
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
});

// Function to toggle information card visibility
function toggleInfoCard(cardType) {
    const cardBody = document.getElementById(`${cardType}-body`);
    const cardArrow = document.getElementById(`${cardType}-arrow`);
    const cardHeader = document.querySelector(`#${cardType}-card .info-card-header`);

    if (cardBody.style.display === 'none') {
        cardBody.style.display = 'block';
        cardArrow.textContent = '▲';
        cardArrow.setAttribute('aria-label', `Collapse ${cardType} explanation`);
        cardHeader.setAttribute('aria-expanded', 'true');
    } else {
        cardBody.style.display = 'none';
        cardArrow.textContent = '▼';
        cardArrow.setAttribute('aria-label', `Expand ${cardType} explanation`);
        cardHeader.setAttribute('aria-expanded', 'false');
    }
}