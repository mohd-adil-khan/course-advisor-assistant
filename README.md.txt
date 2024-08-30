# Course Advisor Assistant

## Overview

The **Course Advisor Assistant** is an AI-powered application designed to help students at IIT Guwahati identify courses offered by different departments that align with their research interests. This tool leverages advanced natural language processing (NLP) techniques, including Retrieval-Augmented Generation (RAG), to analyze course syllabi and provide recommendations based on the similarity between the user's input and the course contents. The application utilizes LangChain, Hugging Face APIs, and Cohere for enhanced document retrieval and contextual compression.

## System Flow

1. The user inputs their research interest in the form of a query.
2. The query is converted into embeddings and matched against the syllabi PDF documents.
3. The most similar sections of the PDFs are supplied as context to the language model.
4. The language model (LLM) responds to the query based on the context of the syllabus.

## Features

- **PDF Course Syllabus Analysis:** The application can parse, understand, and analyze PDF documents containing course syllabi.
- **Contextual Search:** Retrieves the most relevant courses based on the user's research interests using advanced retrieval and reranking techniques.
- **Streamlit Interface:** Provides a user-friendly web interface for interacting with the application.
- **Customizable:** Easily adaptable to other institutions or additional features by modifying the syllabi.

## Installation

### Prerequisites

- Python 3.8 or later
- Install dependencies from the `requirements.txt` file

    ```bash
    pip install -r requirements.txt
    ```

### Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/mohd-adil-khan/course-advisor-assistant.git
    cd course-advisor-assistant
    ```

2. **API Keys**

    Ensure that you have valid API keys for the following services:

    - Hugging Face
    - Cohere

    Set them up in your environment:

    ```bash
    export HUGGINGFACEHUB_API_TOKEN='your_huggingface_token'
    export COHERE_API_KEY='your_cohere_api_key'
    ```

3. **Run the Application**

    To launch the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Usage

1. **Enter Research Domain:**
    - Input your area of interest in the provided text area on the interface.
2. **Get Course Recommendation:**
    - Click on the "Get Course Recommendation" button.
    - The app will process your input and return the most relevant courses based on syllabus similarity.
3. **Review the Results:**
    - The recommended courses will be displayed on the interface. You can review and cross-check these recommendations.

## File Structure

- **app.py**: The main script that runs the Streamlit application.
- **trialsyllabus.pdf**: A sample PDF used for demonstration. You can replace it with your own.
- **IITG_logo.png**: Logo image used in the interface.
- **README.md**: Documentation for the project.

## Contributions

Contributions are welcome! Feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- **Hugging Face:** For the language models and embeddings.
- **Cohere:** For the document compression and reranking tools.
- **LangChain:** For the powerful NLP framework.
- **Streamlit:** For the easy-to-use web application interface.
