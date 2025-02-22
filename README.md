# 🙈 SentinelDocs: AI-Powered Confidential Document Intelligence  
🔐 **SentinelDocs** is a **privacy-first AI assistant** designed to **analyze, summarize, and search confidential documents** with **advanced natural language understanding**. Runs **entirely offline**, ensuring **100% data privacy**. It supports PDFs, DOCX, and TXT files, enabling users to chat with documents, extract key insights, and generate reports—all locally with zero cloud dependency.

---

## 🚀 Features  
✅ **Chat with Documents** – Ask questions.  
✅ **Semantic Search** – Find the most relevant document **by meaning** (not just keywords).  
✅ **AI Summarization** – Generate concise bullet-point summaries.  
✅ **Smart Keyword Extraction** – Detect people, organizations, and dates.  
✅ **Multiple File Support** – Process PDFs, DOCX, and TXT files.  
✅ **Offline & Secure** – Runs **locally** with **zero cloud dependency**.  
✅ **PDF Report Generation** – Export AI-generated insights as reports.  

---

## 🛠️ Tech Stack  
- **Python** (Streamlit, LangChain, FAISS, PyMuPDF, python-docx)  
- **Ollama** (DeepSeek, Mistral) for AI-powered Q&A and summarization  
- **FAISS** for high-performance document search  
- **Sentence-Transformers** for AI embeddings  
- **FPDF** for AI-driven report generation  

---

## **🛠 Setup Instructions**

### **🔹 Option 1: Setup on MacBook Pro (Conda Environment)**

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python3

# Clone the Repository
git clone https://github.com/YOUR_USERNAME/SentinelDocs.git
cd SentinelDocs

# Create a Conda Environment
conda env create -f environment.yml
conda activate sentineldocs

# Run ollama
ollama serve &

# Pull the deepseek-r1:8b model
ollama pull deepseek-r1:8b
ollama pull mistral

# Run the Application
streamlit run app.py

# Or

# Run the app in the background
nohup streamlit run app.py > output.log 2>&1 &
```
## 📝 Example Usage  
1. Upload your confidential documents (PDF, DOCX, TXT)
2. Ask questions and get AI-powered answers
3. View smart summaries & extracted keywords
4. Download AI-generated reports

## 📌 Future Enhancements  
- In-document search – AI finds exact paragraphs answering queries
- GPU acceleration – Faster AI processing with CUDA
- Role-based access – Controlled access for different users

## 💡 Contributing  
Pull requests are welcome! If you'd like to contribute, open an issue or create a PR.

