# 🧠 KnowMap - Cross-Domain Knowledge Mapping Platform

![KnowMap Logo](images/logo.png)

**KnowMap** is an AI-powered platform that helps you discover and visualize relationships across different domains through intelligent knowledge graph technology.

## 🚀 Features

- **Smart Entity Extraction** - Automatically identify entities from text
- **Relationship Discovery** - Find connections between concepts  
- **Semantic Search** - Intelligent search across knowledge graphs
- **Interactive Visualization** - Beautiful graph visualizations
- **Multi-Domain Support** - Medical, scientific, technical, and general knowledge
- **Docker Ready** - Easy deployment with Docker

## 🛠️ Installation

### Quick Start (Colab)
```python
!pip install -q gradio spacy transformers torch pandas networkx pyvis sentence-transformers wikipedia-api requests plotly scikit-learn
!python -m spacy download en_core_web_sm
# Run app.py
