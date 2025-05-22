# FAQ Detection & Response Model

A Node.js application that uses Google's Gemini AI to detect, match, and answer FAQ queries via semantic search.

## Overview

This project implements a sophisticated FAQ bot that:
1. Uses Google's Gemini embeddings API to encode FAQ content and user queries
2. Employs Approximate Nearest Neighbor (ANN) search via HNSW (Hierarchical Navigable Small World) algorithm for efficient similarity matching
3. Leverages Gemini's generative model to produce accurate and contextual answers
4. Includes visualization tools for embedding analysis

## Features

- **High-Performance Vector Search**: Fast similarity matching using the HNSW algorithm
- **Context-Aware Responses**: Analyzes semantic relevance before generating answers
- **Embedding Visualization**: Exports data compatible with TensorFlow Projector for analysis
- **User Query History**: Maintains a record of user interactions and responses
- **Confidence Scoring**: Provides similarity percentages for matched FAQs

## Prerequisites

- Node.js (v14 or higher)
- Google Gemini API key

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd FAQdetectionmodel
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Create a `.env` file in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Prepare your FAQ dataset:
   - Format as a JSON array in `/data/faq_dataset.json`
   - Each FAQ should have `question` and `answer` fields

2. Run the application:
   ```
   node faqdetect.js
   ```

3. Interact with the FAQ bot in the terminal:
   - Type your questions and get AI-generated answers
   - Type "exit" to quit the application

## Visualization

The application exports embedding data to:
- `data/embeddings.tsv`
- `data/metadata.tsv`

These files can be loaded into [TensorFlow Projector](https://projector.tensorflow.org/) for 3D visualization and clustering analysis.

## Project Structure

```
FAQdetectionmodel/
├── faqdetect.js          # Main application code
├── package.json          # Node.js dependencies
├── .env                  # Environment variables (API keys)
├── .gitignore            # Git ignore file
└── data/
    ├── faq_dataset.json  # Your FAQ dataset
    ├── embeddings.tsv    # Exported embeddings for visualization
    ├── metadata.tsv      # Metadata for visualization
    └── user_queries.json # History of user queries and responses
```

## How It Works

1. **Embedding Generation**: Each FAQ item and user query is converted to a high-dimensional vector using Gemini's text-embedding-004 model
2. **ANN Index Building**: An HNSW index is constructed from the FAQ embeddings for efficient similarity search
3. **Query Processing**: When a user asks a question, its embedding is compared to the FAQ embeddings
4. **Answer Generation**: The most relevant FAQ content is passed as context to Gemini's generative model
5. **Response Evaluation**: The model evaluates whether the context truly answers the question before responding

## Advanced Configuration

You can adjust these parameters in `faqdetect.js`:
- `EMBEDDING_MODEL`: The Gemini model used for embeddings
- `GENERATIVE_MODEL`: The Gemini model used for answer generation
- `HNSW_EF_CONSTRUCTION`: Controls index build quality (higher = better, slower)
- `HNSW_M`: Controls graph connectivity (higher = better recall, more memory)
- `HNSW_EF_SEARCH`: Controls search accuracy (higher = better, slower)

## License

[Your License]
