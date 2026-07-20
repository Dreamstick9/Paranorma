# RAG Chatbot

A lightweight chatbot demo that combines vector search with a conversational model.

This project is built around a small mystery-places dataset and uses MongoDB to store vectors for semantic search. The chatbot then uses those search results as context to answer user questions.

## What it does

- Connects to MongoDB using `pymongo`
- Runs a vector search over stored embeddings
- Sends the matched context to a chat completion model
- Returns a natural answer based only on the retrieved context

## How to run

1. Create a `.env` file with the following values:
   - `VOYAGE_API_KEY`
   - `MONGO_URI`
   - `ANTHROPIC_API_KEY`

2. Install dependencies:
   ```bash
   pip install pandas python-dotenv pymongo openai voyageai
   ```

3. Run the script:
   ```bash
   python3 main.py
   ```

## Notes

- The current script prints the model's response for a sample query.
- The dataset and embedding logic are present in the code, but many of those steps are commented out.
- This is a simple, experimental demo for combining retrieval with a generative model.
