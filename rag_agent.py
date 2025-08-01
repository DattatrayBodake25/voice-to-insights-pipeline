from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config
import os


class RAGAgent:
    def __init__(self):
        self.config = Config()
        self._init_components()
    
    def _init_components(self):
        """Initialize RAG components (Vector Store, LLM, Prompt)"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists(self.config.VECTOR_DB_PATH):
            self.vector_store = FAISS.load_local(
                self.config.VECTOR_DB_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self._create_vector_store()

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.config.GEMINI_API_KEY,
            temperature=0.3
        )

        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a customer support assistant. Based on the conversation summary and 
            relevant documents, suggest appropriate follow-up actions.
            
            Conversation Summary:
            {summary}
            
            Relevant Documents:
            {context}
            
            Provide 2-3 specific follow-up actions considering:
            1. Customer's emotional state
            2. Nature of the inquiry
            3. Company policies
            4. Any unresolved issues
            
            Format your response as bullet points.
            """
        )

    def _create_vector_store(self):
        """Create FAISS vector store from local text documents"""
        print("[INFO] Loading documents...")
        loader = DirectoryLoader(self.config.DOCUMENTS_DIR, glob="**/*.txt")
        documents = loader.load()

        if not documents:
            raise ValueError(f"No documents found in {self.config.DOCUMENTS_DIR}")

        print(f"[INFO] Loaded {len(documents)} documents.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        if not splits:
            raise ValueError("Text splitting failed. No chunks generated from documents.")

        print(f"[INFO] Created {len(splits)} chunks for embedding.")

        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.vector_store.save_local(self.config.VECTOR_DB_PATH)
        print(f"[INFO] Vector store saved to {self.config.VECTOR_DB_PATH}")

    def get_followup_actions(self, conversation_summary: str) -> str:
        """Generate follow-up actions using similarity search + Gemini LLM"""
        try:
            docs = self.vector_store.similarity_search(conversation_summary, k=3)
            if not docs:
                raise ValueError("No similar documents found for the given conversation summary.")

            context = "\n\n".join([doc.page_content for doc in docs])

            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "summary": conversation_summary,
                "context": context
            })

            return response.content

        except Exception as e:
            raise RuntimeError(f"[RAG ERROR] {str(e)}")