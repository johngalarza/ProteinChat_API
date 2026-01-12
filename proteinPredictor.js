const ort = require("onnxruntime-node");
const Database = require("better-sqlite3");
const path = require("path");

let scalerSession = null;
let db = null;

function extractFeaturesFast(sequence) {
  const aminoAcids = 'ACDEFGHIKLMNPQRSTVWY';
  const seqLen = sequence.length;
  
  const features = [];
  
  features.push(Math.log1p(seqLen));
  const aaCounts = {};
  aminoAcids.split('').forEach(aa => aaCounts[aa] = 0);
  
  for (const aa of sequence) {
    if (aaCounts.hasOwnProperty(aa)) {
      aaCounts[aa]++;
    }
  }
  
  aminoAcids.split('').forEach(aa => {
    features.push(aaCounts[aa] / seqLen);
  });
  
  const hydrophobic = sequence.split('').filter(aa => 'AILMFVPWG'.includes(aa)).length;
  const positive = sequence.split('').filter(aa => 'KRH'.includes(aa)).length;
  const negative = sequence.split('').filter(aa => 'DE'.includes(aa)).length;
  const polar = sequence.split('').filter(aa => 'STNQ'.includes(aa)).length;
  const aromatic = sequence.split('').filter(aa => 'FWY'.includes(aa)).length;
  const small = sequence.split('').filter(aa => 'AGSV'.includes(aa)).length;
  
  features.push(
    hydrophobic / seqLen,
    positive / seqLen,
    negative / seqLen,
    polar / seqLen,
    aromatic / seqLen,
    small / seqLen
  );
  
  return features;
}

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.pow(a[i] - b[i], 2);
  }
  return Math.sqrt(sum);
}

async function initModel() {
  try {
    console.log("Cargando scaler ONNX...");
    scalerSession = await ort.InferenceSession.create("./models/scaler.onnx");
    console.log("✓ Scaler cargado");
    
    console.log("Conectando a base de datos...");
    const dbPath = path.join(__dirname, 'models', 'protein_index.db');
    
    const fs = require('fs');
    if (!fs.existsSync(dbPath)) {
      throw new Error(`Base de datos no encontrada: ${dbPath}`);
    }
    
    db = new Database(dbPath, { readonly: true, fileMustExist: true });
    
    db.pragma('cache_size = -64000');
    db.pragma('temp_store = memory');
    
    const count = db.prepare('SELECT COUNT(*) as count FROM proteins').get();
    console.log(`✓ Base de datos conectada: ${count.count.toLocaleString()} proteínas`);
    
    return true;
  } catch (error) {
    console.error("Error inicializando modelo:", error);
    throw error;
  }
}

function findNearestNeighbors(scaledFeatures, topN = 5, maxScan = 50000) {
  if (!db) {
    throw new Error("Base de datos no conectada");
  }
  
  const startTime = Date.now();
  
  const stmt = db.prepare('SELECT id, protein_name, sequence, organism, description, features FROM proteins LIMIT ?');
  
  const distances = [];
  
  const proteins = stmt.all(maxScan);
  
  for (const protein of proteins) {
    const proteinFeatures = JSON.parse(protein.features);
    const dist = euclideanDistance(scaledFeatures, proteinFeatures);
    
    distances.push({
      id: protein.id,
      protein_name: protein.protein_name,
      organism: protein.organism,
      description: protein.description,
      sequence: protein.sequence,
      distance: dist
    });
  }
  
  distances.sort((a, b) => a.distance - b.distance);
  const topResults = distances.slice(0, topN);
  
  const searchTime = Date.now() - startTime;
  
  return topResults.map((item, index) => {
    const similarity = Math.max(0, 100 * (1 - item.distance / 10));
    
    return {
      rank: index + 1,
      protein: item.protein_name,
      organism: item.organism,
      description: item.description,
      similarity: similarity.toFixed(2) + '%',
      distance: item.distance.toFixed(4),
      sequence: item.sequence,
      searchTime: index === 0 ? searchTime : undefined
    };
  });
}

function findNearestNeighborsFast(scaledFeatures, topN = 5, inputLength) {
  if (!db) {
    throw new Error("Base de datos no conectada");
  }
  
  const startTime = Date.now();
  
  const minLength = Math.floor(inputLength * 0.8);
  const maxLength = Math.ceil(inputLength * 1.2);
  
  const stmt = db.prepare(`
    SELECT id, protein_name, sequence, organism, description, features 
    FROM proteins 
    WHERE seq_length BETWEEN ? AND ?
    LIMIT 100000
  `);
  
  const proteins = stmt.all(minLength, maxLength);
  console.log(`   Analizando ${proteins.length} proteínas de longitud similar...`);
  
  const distances = [];
  
  for (const protein of proteins) {
    const proteinFeatures = JSON.parse(protein.features);
    const dist = euclideanDistance(scaledFeatures, proteinFeatures);
    
    distances.push({
      id: protein.id,
      protein_name: protein.protein_name,
      organism: protein.organism,
      description: protein.description,
      sequence: protein.sequence,
      distance: dist
    });
  }
  
  distances.sort((a, b) => a.distance - b.distance);
  const topResults = distances.slice(0, topN);
  
  const searchTime = Date.now() - startTime;
  
  return topResults.map((item, index) => {
    const similarity = Math.max(0, 100 * (1 - item.distance / 10));
    
    return {
      rank: index + 1,
      protein: item.protein_name,
      organism: item.organism,
      description: item.description,
      similarity: similarity.toFixed(2) + '%',
      distance: item.distance.toFixed(4),
      sequence: item.sequence,
      searchTime: index === 0 ? searchTime : undefined
    };
  });
}

async function predictProtein(sequence, topN = 5, useFastSearch = true) {
  const totalStart = Date.now();
  
  try {
    const features = extractFeaturesFast(sequence);
    
    const tensor = new ort.Tensor(
      "float32",
      Float32Array.from(features),
      [1, 27]
    );
    
    const feeds = {};
    feeds[scalerSession.inputNames[0]] = tensor;
    
    const output = await scalerSession.run(feeds);
    const scaledFeatures = Array.from(output[scalerSession.outputNames[0]].cpuData);
    
    const results = useFastSearch 
      ? findNearestNeighborsFast(scaledFeatures, topN, sequence.length)
      : findNearestNeighbors(scaledFeatures, topN);
    
    const totalTime = Date.now() - totalStart;
    
    return {
      results,
      time: totalTime,
      searchTime: results[0]?.searchTime,
      inputSequence: sequence.substring(0, 100) + (sequence.length > 100 ? '...' : ''),
      inputLength: sequence.length
    };
    
  } catch (error) {
    console.error("Error en predicción:", error);
    throw error;
  }
}

function closeConnections() {
  if (db) {
    db.close();
    console.log("✓ Base de datos cerrada");
  }
}

async function main() {
  try {
    await initModel();
    
    const testSequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFK";
    
    console.log("\n" + "=".repeat(60));
    console.log("PREDICCIÓN DE PROTEÍNA");
    console.log("=".repeat(60));
    console.log(`\nSecuencia (${testSequence.length} aa):`);
    console.log(testSequence.substring(0, 80) + "...");
    
    const prediction = await predictProtein(testSequence, 5);
    
    console.log(`\n✓ Búsqueda completada en ${prediction.time} ms`);
    console.log(`   (Búsqueda KNN: ${prediction.searchTime} ms)`);
    
    console.log("\n" + "=".repeat(60));
    console.log("PROTEÍNAS MÁS SIMILARES");
    console.log("=".repeat(60));
    
    prediction.results.forEach(res => {
      console.log(`\n#${res.rank} - ${res.protein}`);
      console.log(`   Similaridad: ${res.similarity}`);
      console.log(`   Descripción: ${res.description.substring(0, 60)}...`);
      console.log(`   Organismo: ${res.organism}`);
      console.log(`   Distancia: ${res.distance}`);
    });
    
    console.log("\n" + "=".repeat(60));
    console.log("📋 COMPARACIÓN CON LA MÁS SIMILAR");
    console.log("=".repeat(60));
    console.log(`\nTu secuencia (primeros 100 aa):`);
    console.log(testSequence.substring(0, 100));
    console.log(`\nProteína similar (primeros 100 aa):`);
    console.log(prediction.results[0].sequence.substring(0, 100));
    
  } catch (error) {
    console.error("Error:", error);
  } finally {
    closeConnections();
  }
}

module.exports = {
  initModel,
  predictProtein,
  extractFeaturesFast,
  closeConnections
};

if (require.main === module) {
  main().catch(console.error);
}