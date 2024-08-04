# paperSage: A RAG based Chatbot 🚀

## Your PDF Query Companion 
### Unlock the Knowledge in Your PDFs

**paperSage** is a Retrieval-Augmented Generation (RAG) based Question-Answering (QA) application that allows users to ask any questions from a PDF document. Built from scratch, paperSage leverages advanced embedding models to provide accurate and insightful answers from your documents.

### Live application link: [Link🔗](https://huggingface.co/spaces/NeelPatel31/paperSage)

### Features
- **Upload PDFs**: Easily upload your PDF documents.
- **Efficient Processing**: The application processes the entire PDF and notifies you once it's ready.
- **Ask Questions**: Query the content of your PDF and get instant answers.

### How to Use
1. **Upload the PDF**: Select and upload your PDF document.
2. **Wait for Processing**: The application will process the entire PDF. The time required is proportional to the number of pages in the PDF. Once processing is complete, you will be notified.
3. **Ask Questions**: Start asking any questions from the PDF and receive accurate answers.

### Note
The embedding model is running on CPU, so the processing time depends on the number of pages in the PDF.

### Installation
To install and run paperSage, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/NeelPatel31/paperSage.git
    ```
2. Navigate to the project directory:
    ```bash
    cd paperSage
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Run the application:
    ```bash
    python app.py
    ```
    
### Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

Feel free to adjust any sections as needed. If you have any more details or specific instructions you'd like to include, let me know! 😊
