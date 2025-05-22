require('dotenv').config();
const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const readline = require('readline');
const path = require('path');
const { HierarchicalNSW } = require('hnswlib-node');

// Initialize Gemini AI with your API key
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// File to store user queries for visualization
const USER_QUERIES_FILE = 'data/user_queries.json';
const EMBEDDINGS_FILE = 'data/embeddings.tsv';
const METADATA_FILE = 'data/metadata.tsv';

// --- Constants for Embedding Model ---
const EMBEDDING_MODEL = "text-embedding-004";
const TASK_TYPE_RETRIEVAL_QUERY = "RETRIEVAL_QUERY";
const TASK_TYPE_RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT";

// --- Constant for Generative Model ---
const GENERATIVE_MODEL = "gemini-1.5-flash-latest";

// --- Constants for HNSW Index ---
const HNSW_SPACE = 'cosine'; // Space for similarity calculation
const HNSW_EF_CONSTRUCTION = 200; // Index construction parameter
const HNSW_M = 16; // Index construction parameter
const HNSW_EF_SEARCH = 50; // Search parameter

// Compute cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) {
    return 0;  // Avoid division by zero
  }
  return dot / (normA * normB);
}

// Convert similarity score to percentage
function similarityToPercentage(similarity) {
  // Ensure similarity is within [0, 1] range before converting
  const clampedSimilarity = Math.max(0, Math.min(1, similarity));
  return (clampedSimilarity * 100).toFixed(2) + '%';
}

// Get embeddings for text using Gemini API
async function getEmbedding(text, taskType, title) {
  try {
    console.log(`Getting embedding for ${taskType}:`, text.substring(0, 50) + '...');

    const embedContentRequest = {
        model: EMBEDDING_MODEL,
        content: { parts: [{ text: text }] },
        taskType: taskType,
    };

    if (title && taskType === TASK_TYPE_RETRIEVAL_DOCUMENT) {
        embedContentRequest.title = title;
    }

    const embeddingModel = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });
    const result = await embeddingModel.embedContent(embedContentRequest);

    const embedding = result?.embedding;

    if (!embedding || !embedding.values) {
      console.error('Unexpected embedding response format:', JSON.stringify(result, null, 2));
      throw new Error('Invalid embedding response format');
    }

    console.log('Successfully got embedding with', embedding.values.length, 'dimensions');
    return embedding.values;
  } catch (error) {
    console.error(`Error getting embedding for ${taskType}:`, error);
    if (error.response && error.response.data) {
        console.error('API Error Details:', JSON.stringify(error.response.data, null, 2));
    }
    throw error;
  }
}

// Load FAQ dataset from JSON file
async function loadFAQData() {
  const data = fs.readFileSync('data/faq_dataset.json');
  return JSON.parse(data);
}

// Load previously saved user queries
function loadUserQueries() {
  try {
    if (fs.existsSync(USER_QUERIES_FILE)) {
      const data = fs.readFileSync(USER_QUERIES_FILE);
      return JSON.parse(data);
    }
  } catch (err) {
    console.error('Error loading user queries:', err);
  }
  return [];
}

// Export embeddings to TSV file for TensorFlow Projector
function exportEmbeddings(userQueries) {
  try {
    const dataDir = path.dirname(EMBEDDINGS_FILE);
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    // Filter out queries with invalid embeddings and create TSV content
    const validQueries = userQueries.filter(query => 
      query && query.embedding && Array.isArray(query.embedding) && query.embedding.length > 0
    );

    if (validQueries.length === 0) {
      console.warn('No valid embeddings found to export');
      return;
    }

    const embeddingsContent = validQueries
      .map(query => query.embedding.join('\t'))
      .join('\n');

    fs.writeFileSync(EMBEDDINGS_FILE, embeddingsContent);
    console.log(`Embeddings exported to TSV file for visualization (${validQueries.length} entries).`);
  } catch (err) {
    console.error('Error exporting embeddings:', err);
  }
}

// Export metadata to TSV file for TensorFlow Projector
function exportMetadata(userQueries) {
  try {
    const dataDir = path.dirname(METADATA_FILE);
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    // Filter out queries with invalid embeddings
    const validQueries = userQueries.filter(query => 
      query && query.embedding && Array.isArray(query.embedding) && query.embedding.length > 0
    );

    if (validQueries.length === 0) {
      console.warn('No valid metadata found to export');
      return;
    }

    // Create header for metadata TSV
    const header = ['Question', 'Answer', 'Confidence', 'Type', 'Timestamp'].join('\t');
    
    // Create metadata content
    const metadataContent = validQueries
      .map(query => [
        (query.question || '').replace(/\t/g, ' ').replace(/\n/g, ' '),
        (query.answer || '').replace(/\t/g, ' ').replace(/\n/g, ' '),
        query.confidence || 0,
        query.isUserQuery ? 'User Query' : 'FAQ',
        query.timestamp || new Date().toISOString()
      ].join('\t'))
      .join('\n');

    fs.writeFileSync(METADATA_FILE, `${header}\n${metadataContent}`);
    console.log(`Metadata exported to TSV file for visualization (${validQueries.length} entries).`);
  } catch (err) {
    console.error('Error exporting metadata:', err);
  }
}

// Save user query and its embedding for visualization
function saveUserQuery(query, embedding, generatedAnswer, retrievalConfidence) {
  try {
    let userQueries = loadUserQueries();

    // Validate embedding before saving
    if (!embedding || !Array.isArray(embedding) || embedding.length === 0) {
      console.error('Invalid embedding data - skipping save');
      return;
    }

    userQueries.push({
      question: query,
      answer: generatedAnswer || "Could not generate an answer.",
      confidence: retrievalConfidence,
      embedding: embedding,
      isUserQuery: true,
      timestamp: new Date().toISOString()
    });

    if (userQueries.length > 50) {
      userQueries = userQueries.slice(userQueries.length - 50);
    }

    const dataDir = path.dirname(USER_QUERIES_FILE);
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    fs.writeFileSync(USER_QUERIES_FILE, JSON.stringify(userQueries, null, 2));
    console.log('User query and generated answer saved for visualization.');

    // Export data for TensorFlow Projector
    exportEmbeddings(userQueries);
    exportMetadata(userQueries);
  } catch (err) {
    console.error('Error saving user query:', err);
  }
}

// Compute embeddings for all FAQ questions and build the HNSW index
async function computeFAQEmbeddings(faqData) {
  console.log('Computing embeddings for FAQ questions and building ANN index...');
  const numFaqs = faqData.length;
  if (numFaqs === 0) {
      console.warn("No FAQ data found. Cannot build index.");
      return { index: null, faqData: [] };
  }

  let index = null;
  let dimensions = 0;
  const embeddings = [];
  const labels = [];

  let count = 0;
  let userQueries = loadUserQueries();

  // First, remove any existing FAQ entries from userQueries
  userQueries = userQueries.filter(query => query.isUserQuery === true);

  for (let i = 0; i < numFaqs; i++) {
    const item = faqData[i];
    count++;
    console.log(`Processing FAQ ${count}/${numFaqs}: ${item.question.substring(0, 30)}...`);

    try {
      const embedding = await getEmbedding(
          item.answer,
          TASK_TYPE_RETRIEVAL_DOCUMENT,
          item.question
      );

      if (!index && embedding && embedding.length > 0) {
          dimensions = embedding.length;
          console.log(`Initializing HNSW index with ${dimensions} dimensions.`);
          index = new HierarchicalNSW(HNSW_SPACE, dimensions);
          index.initIndex(numFaqs, HNSW_M, HNSW_EF_CONSTRUCTION); // Max elements, M, efConstruction
      }

      if (index && embedding && embedding.length === dimensions) {
          index.addPoint(embedding, i); // Add embedding with its original index as label
          embeddings.push(embedding); // Keep track for visualization export
          labels.push(i);

          // Add FAQ embeddings to the visualization data
          userQueries.push({
            question: item.question,
            answer: item.answer,
            confidence: 1.0, // FAQ entries have full confidence
            embedding: embedding,
            isUserQuery: false,
            timestamp: new Date().toISOString()
          });
      } else if (embedding && embedding.length !== dimensions) {
          console.error(`Error processing FAQ ${count}: Embedding dimension mismatch (${embedding.length} vs ${dimensions}). Skipping.`);
      } else if (!embedding) {
          console.error(`Error processing FAQ ${count}: Failed to get embedding. Skipping.`);
      }

    } catch (error) {
      console.error(`Error processing FAQ ${count} ("${item.question.substring(0, 30)}..."):`, error);
    }
  }

  if (!index) {
      console.error("Failed to initialize HNSW index. No valid embeddings were processed.");
      return { index: null, faqData: [] };
  }

  console.log(`HNSW index built with ${index.getCurrentCount()} items.`);

  // Save all embeddings (including FAQs) for visualization
  try {
    const dataDir = path.dirname(USER_QUERIES_FILE);
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    fs.writeFileSync(USER_QUERIES_FILE, JSON.stringify(userQueries, null, 2));
    console.log('FAQ embeddings saved for visualization.');

    // Export data for TensorFlow Projector
    exportEmbeddings(userQueries);
    exportMetadata(userQueries);
  } catch (err) {
    console.error('Error saving FAQ embeddings:', err);
  }

  console.log('FAQ embeddings computed and index built successfully.');
  // Return the index and the original faqData array (which maps labels back to content)
  return { index, faqData };
}

// Function to log top matches (updated for ANN results)
function logTopMatch(match, faqData) {
  if (!match || !match.label || !match.distance) {
      console.log("No valid match found.");
      return;
  }
  const faqItem = faqData[match.label];
  const similarity = 1 - match.distance; // Convert distance to similarity for cosine space
  console.log('\nTop match (from ANN search):');
  console.log('-----------------------------');
  console.log(`1. Question: "${faqItem.question}"`);
  console.log(`   Score: ${similarityToPercentage(similarity)} (Distance: ${match.distance.toFixed(4)})`);
  console.log(`   Answer: "${faqItem.answer}"`);
  console.log('-----------------------------');
}

// --- Helper function to create the prompt for the generative model ---
function makePrompt(query, relevantPassage) {
    // Escape potential issues in the passage text for the prompt
    const escapedPassage = relevantPassage ? relevantPassage.replace(/"/g, '""').replace(/\n/g, ' ') : "No passage provided.";

    // Instructions for the model:
    // 1. Evaluate if the PASSAGE is relevant to the QUESTION.
    // 2. If relevant, answer the QUESTION based *only* on the PASSAGE.
    // 3. If not relevant, state that the information wasn't found in the provided context or answer generally if it's a conversational question (like 'who are you?').
    return `You are a helpful and friendly FAQ bot.You should only answer the questions related to the FAQs not anything else

First, critically evaluate if the PASSAGE below is relevant and actually answers the user's QUESTION,then give an interactiv answer that matches users question.

*   If the PASSAGE is *not* relevant to the QUESTION or doesn't contain the answer: State that you couldn't find the specific information in the FAQ context.
*   If the QUESTION is a general conversational query (e.g., "hello", "who are you?"): Respond naturally in your persona as an FAQ bot, ignoring the potentially irrelevant PASSAGE.

Do not use outside knowledge.

QUESTION: "${query}"

PASSAGE: "${escapedPassage}"

ANSWER:`;
}

async function runFAQBot() {
  try {
    console.log('Loading FAQ data and computing embeddings...');
    const initialFaqData = await loadFAQData();
    // Compute embeddings and build the index
    const { index: annIndex, faqData } = await computeFAQEmbeddings(initialFaqData);

    if (!annIndex) {
        console.error("Failed to initialize ANN index. Exiting.");
        return;
    }

    // Set search parameters for the index
    annIndex.setEf(HNSW_EF_SEARCH);
    console.log(`HNSW index ready for search (efSearch = ${HNSW_EF_SEARCH}).`);

    const generativeModel = genAI.getGenerativeModel({ model: GENERATIVE_MODEL });
    console.log(`Initialized generative model: ${GENERATIVE_MODEL}`);

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    console.log('\nGemini-powered FAQ Bot (with ANN) is ready! Type your question or "exit" to quit.');
    console.log('==================================================');

    rl.on('line', async (input) => {
      if (input.trim().toLowerCase() === 'exit') {
        rl.close();
        return;
      }

      console.log(`\n🔍 Processing query: ${input}`);
      console.log('--------------------------------------------------');

      try {
        const queryEmbedding = await getEmbedding(
            input,
            TASK_TYPE_RETRIEVAL_QUERY
        );

        if (!queryEmbedding || queryEmbedding.length === 0) {
            throw new Error("Failed to get embedding for the query.");
        }

        // --- ANN Search ---
        console.log("Performing ANN search...");
        const numNeighbors = 1; // Find the single best match
        const searchResult = annIndex.searchKnn(queryEmbedding, numNeighbors);

        let prompt;
        let retrievalConfidence = 0;
        let bestMatchPassage = null;
        let bestMatchQuestion = null;

        if (searchResult && searchResult.neighbors && searchResult.neighbors.length > 0) {
            const topMatch = {
                label: searchResult.neighbors[0], // Label is the original index in faqData
                distance: searchResult.distances[0]
            };
            const bestMatchIndex = topMatch.label;

            if (bestMatchIndex >= 0 && bestMatchIndex < faqData.length) {
                const matchedFaq = faqData[bestMatchIndex];
                retrievalConfidence = 1 - topMatch.distance; // Convert distance to similarity
                bestMatchPassage = matchedFaq.answer;
                bestMatchQuestion = matchedFaq.question;

                logTopMatch(topMatch, faqData); // Log the best match found by ANN

                console.log(`\nℹ️ Providing best match (Similarity: ${similarityToPercentage(retrievalConfidence)}) as context to the model.`);
                console.log(`   Q: ${bestMatchQuestion}`);
            } else {
                 console.warn(`\n⚠️ ANN search returned invalid index: ${bestMatchIndex}`);
                 bestMatchPassage = null; // No valid passage found
            }

        } else {
            console.log("\n⚠️ ANN search did not return any neighbors.");
            bestMatchPassage = null; // No passage found
        }

        prompt = makePrompt(input, bestMatchPassage);

        let generatedAnswer = "Sorry, I encountered an issue generating the answer.";
        console.log('\n🧠 Generating final answer (model will evaluate context relevance)...');

        try {
            const result = await generativeModel.generateContent(prompt);
            const response = result?.response;
            const text = response?.text();

            if (text) {
                generatedAnswer = text.trim();
            } else {
                console.error("Failed to get text from generative model response:", JSON.stringify(response, null, 2));
            }
        } catch (genError) {
            console.error('Error generating content with generative model:', genError);
            if (genError.response && genError.response.data) {
                console.error('API Error Details:', JSON.stringify(genError.response.data, null, 2));
            }
        }

        console.log('\n💬 Answer:');
        console.log(generatedAnswer);
        console.log('--------------------------------------------------');

        // Save user query with its embedding and the ANN-retrieved confidence
        saveUserQuery(input, queryEmbedding, generatedAnswer, retrievalConfidence);

      } catch (error) {
        console.error('Error processing query:', error);
        // Attempt to save query even if processing failed, maybe without embedding/confidence
        saveUserQuery(input, [], "Error processing query.", 0);
      }

      console.log('\nType another question or "exit" to quit.');
      console.log('==================================================');
    });
  } catch (error) {
    console.error('Error running the FAQ bot:', error);
  }
}

// Run the FAQ bot
runFAQBot().catch(err => {
  console.error('Fatal error starting the FAQ bot:', err);
}); 
