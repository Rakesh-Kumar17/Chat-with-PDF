
# Chat with PDF Application
A Streamlit application that enables conversational interaction with PDF documents using Google's Gemini AI. The app highlights relevant sections in the PDF that match the AI's responses.

## Features
- PDF document upload and processing
- Natural language querying of PDF content
- Auto-highlighting of relevant text segments
- Interactive PDF preview
- Downloadable highlighted PDFs
- Multi-PDF support

## Prerequisites
-Python 3.8+
-Google Cloud API key
## Installation
1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/pdf-chat-assistant.git
   
   cd pdf-chat-assistant
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Create .env file and add your Google API key
   ```bash
   GOOGLE_API_KEY=your_api_key_here
## Required Packages
- streamlit
- PyPDF2
- langchain
- google-generativeai
- python-dotenv
- faiss-cpu
- PyMuPDF

## Start the application:

streamlit run app.py

## Upload PDFs:
-Click "Upload your PDF Files" in sidebar
-Select one or more PDFs
-Click "Submit & Process"

Ask questions about your documents in the text input field
## View results:
Read AI response
See highlighted relevant sections
Download highlighted PDF
View page previews


