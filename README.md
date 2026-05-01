# Amazon Grocery Review IR System

## CAP 6776 — Information Retrieval | Spring 2026

## Overview

An Information Retrieval system built on 10,000 Amazon US Grocery reviews.
Implements BM25, Dense Retrieval (FAISS + Sentence Transformers), Hybrid Search
(Reciprocal Rank Fusion), and RAG (Retrieval-Augmented Generation via Groq LLaMA).

## Setup Instructions

⚠️ **Important: Corpus**

The raw Kaggle TSV file (2.4M records, \~700MB)

(https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?select=amazon\_reviews\_us\_Grocery\_v1\_00.tsv)

is NOT included in this submission. The pre-sampled corpus.csv (10,000 reviews)

is already provided in data/ and is what the notebook uses.



Do NOT re-run Code Cell 1 (Corpus Construction) — it requires the raw TSV file

and will produce a different sample, breaking compatibility with the provided

qrels.csv relevance judgments.



Start from Code Cell 2 onwards installing the required libraries and using the provided corpus.csv from next Code Cell onwards.

### Requirements

pip install rank\_bm25 sentence-transformers faiss-cpu groq pandas numpy nltk

### Running the System

1. Upload data/corpus.csv to your Colab environment
2. Set your GROQ\_API\_KEY in Colab Secrets (Settings → Secrets)
3. Run all cells in order from top to bottom

## File Structure

code/  
notebook.ipynb       - Main notebook (all steps)  
requirements.txt     - Required libraries  
  
../README.md            - This file  
  
data/  
corpus.csv           - 10,000 Amazon Grocery reviews  
queries.csv          - 20 evaluation queries  
qrels.csv            - Relevance judgments (graded 0/1/2)  
  
outputs/  
run\_bm25.csv         - BM25 top-10 results for all 20 queries  
run\_dense.csv        - Dense retrieval top-10 results  
run\_hybrid.csv       - Hybrid (RRF) top-10 results  
metrics\_summary.csv  - P@10, MAP, nDCG@10 per query per method  
rag\_outputs.csv      - RAG answers for all 20 queries  
pr\_curve.png         - Precision-Recall curve (q01)  
  
walkthrough/  
walkthrough.pdf      - Annotated screenshots  
  
## System Architecture  
  
1. BM25 (sparse)         — keyword-based retrieval using rank\_bm25  
2. Dense Retrieval        — all-MiniLM-L6-v2 embeddings + FAISS index  
3. Hybrid Search (Bonus)  — Reciprocal Rank Fusion (RRF, k=60) of BM25 + Dense  
4. RAG (Bonus)            — Top-3 hybrid results as context for LLaMA-3.3-70B via Groq  
  
## Evaluation Results (averaged over 20 queries)  
  
|Method|P@10|MAP|nDCG@10|
|-|-|-|-|
|BM25|0.295|0.2712|0.3980|
|Dense|0.285|0.3416|0.4738|
|Hybrid|0.585|0.6786|0.7468|

## Walkthrough  
  
See walkthrough/walkthrough.pdf  
  
## AI Tool Disclosure  
  
Claude (Anthropic) was used as a coding aid during development.  
All code has been reviewed, tested, and understood by the author.  
The paper is written in the author's own voice.  
  
## Bonus Technique  
  
Hybrid Search (BM25 + Dense via RRF) — explored in HW5 summary.  
RAG pipeline using Groq LLaMA-3.3-70B grounded on retrieved reviews.

