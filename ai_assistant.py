import os
import shutil
import openai
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import time

# Charger les variables d'environnement
load_dotenv()

# Initialiser l'état global pour les feedbacks
if "feedback_answer" not in st.session_state:
    st.session_state.feedback_answer = "default"
if "feedback_question" not in st.session_state:
    st.session_state.feedback_question = ""

# CSS pour ajuster la taille et le style de certains éléments
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .response-text {
        font-size: 1.5em;
        color: white;
        font-weight: bold;
        
    }
    .big-font {
        font-size: 1.2em !important;
        font-weight: bold;
    }
    .css-1cpxqw2 {  /* Style pour les boutons Streamlit */
        font-size: 1.1em;
        font-weight: bold;
    }
    .css-16idsys {  /* Style pour les titres de la question */
        font-size: 1.1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def set_background(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{get_base64_image(image_path)});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def get_base64_image(image_file):
    import base64
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

openai.api_key = os.getenv('OPENAI_API_KEY')

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

PROMPT_TEMPLATE = """
Answer the question based only on the following context. :

{context}

---

Answer the question based on the above context: {question}
"""

class DocumentProcessor:
    def __init__(self, data_path, chroma_path):
        self.data_path = data_path
        self.chroma_path = chroma_path

    def generate_data_store(self):
        documents = self.load_documents()
        if not documents:
            st.error("Aucun document n'a été chargé. Assurez-vous que les fichiers sont valides et que le répertoire de données est correct.")
            return
        chunks = self.split_text(documents)
        if not chunks:
            st.error("Aucun chunk de texte n'a été généré. Vérifiez que les documents contiennent du texte.")
            return
        self.save_to_chroma(chunks)

    def load_documents(self):
        all_files = os.listdir(self.data_path)
        print("Contenu de DATA_PATH :", all_files)
        documents = []
        
        for file_name in os.listdir(self.data_path):
            if file_name.endswith(".pdf"):
                pdf_loader = PyMuPDFLoader(os.path.join(self.data_path, file_name))
                pdf_documents = pdf_loader.load()
                documents.extend(pdf_documents)
        
        txt_loader = DirectoryLoader(self.data_path, glob="*.txt", loader_cls=TextLoader)
        txt_documents = txt_loader.load()
        documents.extend(txt_documents)

        md_loader = DirectoryLoader(self.data_path, glob="*.md", loader_cls=TextLoader)
        md_documents = md_loader.load()
        documents.extend(md_documents)

        if not documents:
            print("Aucun document valide trouvé dans le répertoire de données.")
        else:
            print(f"{len(documents)} documents ont été chargés avec succès.")

        return documents

    def split_text(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        
        return chunks

    def save_to_chroma(self, chunks: list[Document]):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=self.chroma_path
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {self.chroma_path}.")

    def answer_question(self, question):
        db = Chroma(persist_directory=self.chroma_path, embedding_function=OpenAIEmbeddings())
        results = db.similarity_search_with_relevance_scores(question, k=9)
        if len(results) == 0 or results[0][1] < 0.5:
            return "Impossible de trouver des résultats correspondants."

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=question)
        model = ChatOpenAI()
        response = model.invoke(prompt)
        
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response.content}"

        return formatted_response


def main():
    st.title("Your Personalized AI Assistant .")
    
    background_image_path = "bg.jpg"
    set_background(background_image_path)
    
    logo_path = "logo.png"
    st.image(logo_path, width=500)
    
    st.markdown('<div class="big-font">Upload your files (PDF, TXT, MD)</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type=["pdf", "txt", "md"], accept_multiple_files=True)
    
    if st.button("Gérer la base de données"):
        if uploaded_files:
            try:
                if not os.path.exists(DATA_PATH):
                    os.makedirs(DATA_PATH)
                
                for uploaded_file in uploaded_files:
                    with open(os.path.join(DATA_PATH, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())

                processor = DocumentProcessor(DATA_PATH, CHROMA_PATH)
                processor.generate_data_store()
                
                st.success("Base de données créée avec succès.")
            except Exception as e:
                st.error(f"Erreur lors de la création de la base de données : {e}")
        else:
            st.warning("Veuillez télécharger au moins un fichier.")

    st.markdown('<div class="big-font">Ask your question:</div>', unsafe_allow_html=True)
    question = st.text_input("")
    st.session_state.feedback_question = question
    
    if "show_feedback" not in st.session_state:
        st.session_state.show_feedback = False
     
    if st.button("Obtenir une réponse"):
        if question:
            processor = DocumentProcessor(DATA_PATH, CHROMA_PATH)

            with st.spinner("Génération de la réponse..."):
                answer = processor.answer_question(question)
                st.session_state.feedback_answer = answer
                
                print("la reponse est : ", answer)

            if isinstance(answer, str):
                words = answer.split()
                response_text = ""
                response_container = st.empty()

                for i, word in enumerate(words):
                    response_text += word + " "
                    
                    if i % 5 == 0:
                        response_container.markdown(
                            f"<div class='response-text'>{response_text}</div>",
                            unsafe_allow_html=True
                        )
                        time.sleep(0.1)

                st.balloons()
                st.session_state.show_feedback = True
            else:
                st.error("La réponse n'est pas au format attendu.")
        else:
            st.warning("Veuillez poser une question.")
           
    print("before condition : ", st.session_state.feedback_answer)
    
    if st.session_state.show_feedback:
        st.markdown("### Votre feedback sur la réponse")
        rating = st.slider("Notez cette réponse", 1, 5, 3)
        feedback = st.text_area("Commentaires supplémentaires")
        
        if st.button("Soumettre le feedback"):
            with open("feedback.csv", "a") as f:
                f.write(f"{st.session_state.feedback_answer},{st.session_state.feedback_question},{rating},{feedback}\n")
            st.success("Merci pour votre feedback!")
            st.session_state.show_feedback = False

if __name__ == "__main__":
    main()
