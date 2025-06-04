const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const pdfParse = require('pdf-parse');
const cors = require('cors');

const app = express();
const port = 3000;

// Set up middleware
app.use(express.json());
app.use(cors());
app.use(express.static('public'));

// Configure multer for PDF uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'uploads';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype !== 'application/pdf') {
      return cb(new Error('Only PDF files are allowed'));
    }
    cb(null, true);
  }
});

// Python server URL (FAISS and embedding service)
const PYTHON_SERVER_URL = 'http://localhost:5000';

// 1. PDF Upload and Processing
app.post('/api/upload', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No PDF file uploaded' });
    }

    const filePath = req.file.path;
    
    // Read the PDF
    const dataBuffer = fs.readFileSync(filePath);
    const pdfData = await pdfParse(dataBuffer);
    const text = pdfData.text;

    // Send to Python server for processing and embedding
    const response = await axios.post(`${PYTHON_SERVER_URL}/process`, {
      text,
      filename: path.basename(filePath)
    });

    res.json({
      success: true,
      message: 'PDF processed successfully',
      documentId: response.data.document_id,
      filename: req.file.originalname,
      pageCount: pdfData.numpages
    });
  } catch (error) {
    console.error('Error processing PDF:', error);
    res.status(500).json({ error: 'Failed to process PDF', details: error.message });
  }
});

// 2. Chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { query, documentId } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'No query provided' });
    }

    // Retrieve relevant document chunks from FAISS via Python server
    const retrievalResponse = await axios.post(`${PYTHON_SERVER_URL}/retrieve`, {
      query,
      document_id: documentId
    });

    // Get relevant chunks
    const { contexts } = retrievalResponse.data;

    // Generate response using Ollama and the retrieved context
    const generateResponse = await axios.post(`${PYTHON_SERVER_URL}/generate`, {
      query,
      contexts
    });

    res.json({
      answer: generateResponse.data.response,
      contexts: contexts.map(c => ({ text: c.text, score: c.score }))
    });
  } catch (error) {
    console.error('Error during chat:', error);
    res.status(500).json({ error: 'Failed to generate response', details: error.message });
  }
});

// 3. List available documents
app.get('/api/documents', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/documents`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching documents:', error);
    res.status(500).json({ error: 'Failed to fetch documents', details: error.message });
  }
});

// 4. Delete document
app.delete('/api/documents/:id', async (req, res) => {
  try {
    const docId = req.params.id;
    const response = await axios.delete(`${PYTHON_SERVER_URL}/documents/${docId}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error deleting document:', error);
    res.status(500).json({ error: 'Failed to delete document', details: error.message });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`RAG Chatbot server running at http://localhost:${port}`);
});