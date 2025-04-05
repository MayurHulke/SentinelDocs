# 🙈 SentinelDocs

**Your Private AI-Powered Document Analyst**

SentinelDocs is a privacy-focused document analysis tool that leverages local AI models to analyze your documents without sending data to external services.

## Features

- **📂 Multi-format Document Support**: Upload and process PDF, DOCX, and TXT files
- **🔍 Semantic Search**: Ask questions about your documents and get accurate answers
- **📊 Document Statistics**: View word count, character count, sentences, and key entities
- **🧠 AI-Powered Analysis**: Uses Ollama-based local LLMs to analyze your documents
- **🔄 Cross-Document Comparison**: Compare multiple documents to identify similarities and differences
- **📱 Clean, Responsive UI**: Built with Streamlit for a modern and accessible interface
- **📈 Entity Recognition**: Automatically identifies and extracts key entities (people, organizations, locations, etc.)
- **📄 PDF Report Generation**: Create and download comprehensive insights reports
- **🧩 Model Selection**: Choose from available Ollama models for different analysis needs
- **🔐 Privacy Focused**: All processing happens locally, with no data sent to external APIs

## Requirements

- Python 3.8+
- Ollama with LLM models installed (e.g., deepseek-r1:8b)
- Required Python packages (see installation)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SentinelDocs.git
   cd SentinelDocs
   ```

2. Install the required packages:
   ```bash
   pip install streamlit langchain-ollama fpdf faiss-cpu PyMuPDF python-docx spacy sentence-transformers
   ```

3. Download the required spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Install and start Ollama:
   ```bash
   # Follow instructions at https://ollama.ai to install Ollama
   # Then pull a model:
   ollama pull deepseek-r1:8b
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Upload your documents using the file uploader

4. View document statistics and extracted content

5. Compare documents (if multiple are uploaded)

6. Ask questions about your documents using the query interface

7. Generate and download insights reports

## How It Works

1. **Document Processing**: Extracts text from uploaded files
2. **Document Indexing**: Creates a semantic index using FAISS for efficient retrieval
3. **NLP Analysis**: Uses spaCy for entity recognition and basic document statistics
4. **Semantic Search**: When you ask a question, finds the most relevant document passages
5. **AI Response Generation**: Uses the Ollama LLM to generate responses based on the relevant context

## Customization

- **Change Default Model**: Select your preferred model from the dropdown in the sidebar
- **Adjust Chunk Size**: Modify the `chunk_size` parameter in the code for different document segmentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

