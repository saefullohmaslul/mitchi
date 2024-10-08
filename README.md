---
title: Mitchi
emoji: 🐢
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
license: mit
short_description: Smart Research Discussion Chatbot
---

# Mitchi: Smart Research Discussion Chatbot

**Mitchi** is a smart chatbot designed to facilitate research discussions by retrieving and summarizing scientific papers in real-time. By leveraging LLaMA and the arXiv API, Mitchi provides an intuitive way to explore research topics and get insightful paper recommendations, helping researchers dive deep into academic subjects.

## Features:
- **Real-time Retrieval**: Fetch scientific articles based on user queries directly from the arXiv database.
- **LLM-Powered Summaries**: Generate concise and relevant summaries of retrieved papers using LLaMA.
- **Interactive Interface**: Easy-to-use web interface built with Gradio for seamless user interaction.
- **Public Data Source**: Utilizes the arXiv API for open and up-to-date scientific articles.

## Tech Stack:
- **Python**: Core language used for backend development.
- **LLaMA**: Meta's large language model for handling natural language understanding and generation.
- **Grook**: A microservice management framework to handle backend operations.
- **arXiv API**: External source for fetching scientific paper metadata.
- **Gradio**: Web interface for interacting with the chatbot.
- **Huggingface**: Used for LLaMA model integration.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/mitchi.git
   cd mitchi
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory and add the following information:
     ```bash
     ARXIV_API_KEY=your_arxiv_api_key
     ```
   - Replace `your_arxiv_api_key` with your arXiv API key.

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Access the chatbot UI**:
   - The chatbot will be available at `http://localhost:7860` through the Gradio interface.

## Usage
- Enter a research topic, and Mitchi will fetch and summarize relevant articles from the arXiv database.
- Mitchi uses RAG (Retrieval-Augmented Generation) to provide meaningful insights into your queries.

## File Structure
```
mitchi/
│
├── app.py                 # Main application file
├── chatbot/               # Logic for LLaMA and RAG integration
│   ├── retriever.py       # Code for retrieving data from arXiv API
│   ├── summarizer.py      # Code for generating summaries using LLaMA
├── templates/             # HTML templates for the Gradio web interface
├── static/                # Static files (CSS, JS)
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Roadmap
- [x] Integrate arXiv API for paper retrieval.
- [x] Implement LLaMA for query processing and summarization.
- [x] Build user-friendly Gradio interface for interaction.
- [ ] Add support for multiple research paper sources.
- [ ] Improve summarization accuracy for complex queries.

## Contributions
We welcome contributions! Please submit a pull request with detailed information about your changes, and we will review it as soon as possible.

## License
This project is licensed under the MIT License.

---

With Mitchi, academic researchers and curious minds alike can explore scientific topics efficiently, supported by the power of cutting-edge AI technology.
