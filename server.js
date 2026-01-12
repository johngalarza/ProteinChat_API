const express = require('express');
const cors = require('cors');
const { initModel, predictProtein, closeConnections } = require('./proteinPredictor');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

let modelReady = false;

(async () => {
  try {
    console.log("Inicializando modelo...");
    await initModel();
    modelReady = true;
    console.log("✓ Modelo listo para predicciones");
  } catch (error) {
    console.error("❌ Error inicializando modelo:", error);
    process.exit(1);
  }
})();

app.get('/api/health', (req, res) => {
  res.json({
    status: modelReady ? 'ready' : 'loading',
    timestamp: new Date().toISOString()
  });
});

app.post('/api/predict', async (req, res) => {
  if (!modelReady) {
    return res.status(503).json({
      error: 'Modelo aún no está listo',
      message: 'Por favor espera unos segundos e intenta nuevamente'
    });
  }

  try {
    const { sequence, topN = 5 } = req.body;

    if (!sequence || typeof sequence !== 'string') {
      return res.status(400).json({
        error: 'Secuencia inválida',
        message: 'Debes proporcionar una secuencia de proteína válida'
      });
    }

    const cleanSequence = sequence.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
    
    if (cleanSequence.length === 0) {
      return res.status(400).json({
        error: 'Secuencia vacía',
        message: 'La secuencia no contiene aminoácidos válidos'
      });
    }

    if (cleanSequence.length < 10) {
      return res.status(400).json({
        error: 'Secuencia muy corta',
        message: 'La secuencia debe tener al menos 10 aminoácidos'
      });
    }

    const prediction = await predictProtein(cleanSequence, topN);

    res.json({
      success: true,
      data: {
        inputSequence: {
          original: sequence.substring(0, 100) + (sequence.length > 100 ? '...' : ''),
          cleaned: cleanSequence.substring(0, 100) + (cleanSequence.length > 100 ? '...' : ''),
          length: cleanSequence.length
        },
        predictions: prediction.results,
        metadata: {
          processingTime: prediction.time,
          timestamp: new Date().toISOString()
        }
      }
    });

  } catch (error) {
    console.error("Error en predicción:", error);
    res.status(500).json({
      error: 'Error interno',
      message: error.message
    });
  }
});

app.get('/api/model-info', (req, res) => {
  res.json({
    modelName: 'Protein Similarity Predictor',
    version: '1.0.0',
    features: 27,
    algorithm: 'K-Nearest Neighbors',
    database: 'UniProt SwissProt',
    totalProteins: 573661,
    capabilities: [
      'Búsqueda de proteínas similares',
      'Clasificación por similaridad',
      'Análisis de propiedades fisicoquímicas'
    ]
  });
});

app.get('/api/examples', (req, res) => {
  res.json({
    examples: [
      {
        name: 'Hemoglobina Humana (Alpha)',
        sequence: 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFK',
        expectedMatch: 'P69905 (HBA_HUMAN)'
      },
      {
        name: 'Insulina Humana',
        sequence: 'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN',
        expectedMatch: 'P01308 (INS_HUMAN)'
      },
      {
        name: 'Proteína corta (ejemplo mínimo)',
        sequence: 'ACDEFGHIKLMNPQRSTVWY',
        expectedMatch: 'Varias coincidencias posibles'
      }
    ]
  });
});

app.use((req, res) => {
  res.status(404).json({
    error: 'Endpoint no encontrado',
    availableEndpoints: [
      'GET /api/health',
      'POST /api/predict',
      'GET /api/model-info',
      'GET /api/examples'
    ]
  });
});

const server = app.listen(PORT, () => {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`🧬 Servidor de Predicción de Proteínas`);
  console.log(`${'='.repeat(60)}`);
  console.log(`\n✓ Servidor corriendo en http://localhost:${PORT}`);
  console.log(`\nEndpoints disponibles:`);
  console.log(`  - GET  /api/health      - Estado del servidor`);
  console.log(`  - POST /api/predict     - Predicción de proteínas`);
  console.log(`  - GET  /api/model-info  - Información del modelo`);
  console.log(`  - GET  /api/examples    - Ejemplos de uso`);
  console.log(`\n${'='.repeat(60)}\n`);
});

process.on('SIGINT', () => {
  console.log('\n\nCerrando servidor...');
  closeConnections();
  server.close(() => {
    console.log('✓ Servidor cerrado');
    process.exit(0);
  });
});

module.exports = app;