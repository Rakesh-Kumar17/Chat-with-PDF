import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz
import tempfile
import re
from difflib import SequenceMatcher

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    page_texts = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += page_text
            page_texts.append((i, page_text))
    return text, page_texts

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def clean_text(text):
    """Clean text while preserving sentence structure"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sentences(text):
    """Simple sentence splitter using common punctuation"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def find_matching_segments(pdf_text, response_text, threshold=0.7):
    """Find matching segments between PDF and response"""
    matches = []
    
    # Split texts into sentences
    pdf_sentences = split_into_sentences(pdf_text)
    response_sentences = split_into_sentences(response_text)
    
    # Find matching sentences
    for pdf_sent in pdf_sentences:
        pdf_sent = clean_text(pdf_sent)
        if not pdf_sent:
            continue
            
        best_match_score = 0
        
        for resp_sent in response_sentences:
            resp_sent = clean_text(resp_sent)
            if not resp_sent:
                continue
            
            # Calculate similarity score
            similarity = SequenceMatcher(None, pdf_sent.lower(), resp_sent.lower()).ratio()
            
            if similarity > best_match_score:
                best_match_score = similarity
        
        if best_match_score >= threshold:
            matches.append(pdf_sent)
    
    return matches

def highlight_pdf(pdf_path, response_text):
    doc = fitz.open(pdf_path)
    highlighted_segments = []
    temp_path = "highlighted_temp.pdf"
    
    for page in doc:
        page_text = page.get_text()
        matching_segments = find_matching_segments(page_text, response_text)
        
        for segment in matching_segments:
            # Search for the segment in the page
            instances = page.search_for(segment)
            if instances:
                highlighted_segments.append(segment)
                for inst in instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors({"stroke": (1, 1, 0), "fill": (1, 1, 0)})  # Yellow
                    highlight.set_opacity(0.3)
                    highlight.update()

    doc.save(temp_path)
    doc.close()
    return temp_path, highlighted_segments

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, pdf_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    response_text = response["output_text"]
    st.write("Reply: ", response_text)

    if pdf_docs:
        highlighted_pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_docs[0].getvalue())
                tmp_path = tmp_file.name

                highlighted_pdf_path, highlighted_segments = highlight_pdf(tmp_path, response_text)
                
                if highlighted_segments:
                    st.success(f"Highlighted {len(highlighted_segments)} matching segments in the PDF.")
                    st.info("Matching segments:")
                    for i, segment in enumerate(highlighted_segments, 1):
                        st.write(f"{i}. {segment}")
                else:
                    st.warning("No matching segments found between the response and the PDF.")
                
                with open(highlighted_pdf_path, "rb") as file:
                    pdf_bytes = file.read()
                    st.download_button(
                        label="Download Highlighted PDF",
                        data=pdf_bytes,
                        file_name="highlighted_document.pdf",
                        mime="application/pdf"
                    )
                
                with fitz.open(highlighted_pdf_path) as doc:
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_bytes = pix.tobytes()
                        st.image(img_bytes)

                os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"Error in highlighting: {str(e)}")
        
        finally:
            if highlighted_pdf_path and os.path.exists(highlighted_pdf_path):
                os.unlink(highlighted_pdf_path)

def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("Chat with PDF 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text, page_texts = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state['pdf_docs'] = pdf_docs
                st.success("Done")

    if user_question:
        user_input(user_question, st.session_state.get('pdf_docs', None))

if __name__ == "__main__":
    main()