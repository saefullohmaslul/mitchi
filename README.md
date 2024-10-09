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

Mitchi is an intelligent chatbot designed to facilitate efficient research discussions and explorations. Leveraging cutting-edge AI technologies, Mitchi provides users with concise summaries and real-time access to scientific articles, enhancing the way we engage with academic content.

## Features

- **Real-time Retrieval**: Fetch scientific articles based on user queries directly from the arXiv database.
- **LLM-Powered Summaries**: Generate concise and relevant summaries of retrieved papers using LLaMA.
- **Interactive Interface**: Easy-to-use web interface built with Gradio for seamless user interaction.
- **Public Data Source**: Utilizes the arXiv API for open and up-to-date scientific articles.

## Technology Stack

- **Python**: Core language used for backend development.
- **LLaMA**: Meta's large language model for handling natural language understanding and generation.
- **Groq**: A accelerator for LLaMA model.
- **arXiv API**: External source for fetching scientific paper metadata.
- **Gradio**: Web interface for interacting with the chatbot.
- **Huggingface**: Used for LLaMA model integration.

## Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/saefullohmaslul/mitchi.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd mitchi
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**

   ```bash
   python app.py
   ```

## Contributions

We welcome contributions! Please submit a pull request with detailed information about your changes, and we will review it as soon as possible.

## License

This project is licensed under the MIT License.

With Mitchi, academic researchers and curious minds alike can explore scientific topics efficiently, supported by the power of cutting-edge AI technology.
