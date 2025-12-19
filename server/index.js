import express from 'express';
import cors from 'cors';
import multer from 'multer';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Azure Function endpoint
const AZURE_FUNCTION_URL = process.env.AZURE_FUNCTION_URL ||
    'https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/imagedeblur';

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
    storage,
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed'));
        }
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Server is running' });
});

// Image deblurring endpoint
app.post('/api/deblur', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        console.log(`Processing image: ${req.file.originalname} (${(req.file.size / 1024 / 1024).toFixed(2)} MB)`);

        // Convert image buffer to base64 (Azure Function expects plain base64, not data URI)
        const base64Image = req.file.buffer.toString('base64');

        // Send to Azure Function
        const azureResponse = await fetch(AZURE_FUNCTION_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: base64Image
            }),
        });

        if (!azureResponse.ok) {
            const errorText = await azureResponse.text();
            console.error('Azure Function error:', errorText);
            throw new Error(`Azure Function returned ${azureResponse.status}: ${errorText}`);
        }

        const result = await azureResponse.json();

        console.log('Image processed successfully');

        res.json({
            success: true,
            deblurred_image: result.deblurred_image,
            processing_time: result.processing_time
        });

    } catch (error) {
        console.error('Error processing image:', error);
        res.status(500).json({
            error: error.message || 'Failed to process image',
            details: process.env.NODE_ENV === 'development' ? error.stack : undefined
        });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({
        error: err.message || 'Internal server error'
    });
});

app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    console.log(`📡 Azure Function endpoint: ${AZURE_FUNCTION_URL}`);
});
