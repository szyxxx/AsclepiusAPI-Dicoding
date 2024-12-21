import express from 'express';
import { Firestore } from '@google-cloud/firestore';
import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import multer from 'multer';
import { Storage } from '@google-cloud/storage';
import cors from 'cors';

const app = express();
const firestore = new Firestore();
const port = process.env.PORT || 8080;

app.use(cors({
    origin: '*', 
    methods: ['GET', 'POST'], 
    allowedHeaders: ['Content-Type'], 
}));

const upload = multer({ dest: 'uploads/' });

const bucketName = 'submissionmlgc-axeldavid';
const modelFileName = 'model.json';

const modelPublicUrl = `https://storage.googleapis.com/${bucketName}/models/${modelFileName}`;

let model;

async function loadModel() {
    try {
        model = await tf.loadGraphModel(modelPublicUrl);
        console.log('Model loaded successfully from public URL.');
    } catch (error) {
        console.error('Error loading model:', error);
        process.exit(1);
    }
}

loadModel();

app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                status: "fail",
                message: "Tidak ada file gambar yang diunggah"
            });
        }

        const fileSize = req.file.size;
        if (fileSize > 1000000) {
            return res.status(413).json({
                status: "fail",
                message: "Payload content length greater than maximum allowed: 1000000"
            });
        }

        const imageBuffer = fs.readFileSync(req.file.path);
        const imageTensor = tf.node.decodeImage(imageBuffer, 3)
            .resizeBilinear([224, 224])
            .toFloat()
            .div(tf.scalar(255))
            .expandDims(0);

        console.log('Image tensor shape:', imageTensor.shape);

        const prediction = model.predict(imageTensor);
        const predictionData = prediction.dataSync();
        console.log('Raw prediction data:', predictionData);

        const result = predictionData[0] >= 0.58 ? 'Cancer' : 'Non-cancer';
        const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';
        const createdAt = new Date().toISOString();
        const id = uuidv4();

        await firestore.collection('predictions').doc(id).set({
            id,
            result,
            suggestion,
            createdAt
        });

        res.status(201).json({
            status: "success",
            message: "Model is predicted successfully",
            data: {
                id,
                result,
                suggestion,
                createdAt
            }
        });
    } catch (error) {
        console.error('Error during prediction:', error);
        res.status(400).json({
            status: "fail",
            message: "Terjadi kesalahan dalam melakukan prediksi"
        });
    } finally {
        if (req.file) {
            fs.unlinkSync(req.file.path);
        }
    }
});

app.get('/predict/histories', async (req, res) => {
    try {
        const snapshot = await firestore.collection('predictions').get();
        const histories = snapshot.docs.map(doc => ({
            id: doc.id,
            history: {
                ...doc.data(),
                id: doc.id
            }
        }));
        res.json({
            status: "success",
            data: histories
        });
    } catch (error) {
        res.status(500).json({
            status: "fail",
            message: "Terjadi kesalahan saat mengambil riwayat prediksi"
        });
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
