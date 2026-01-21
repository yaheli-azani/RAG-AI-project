import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Use this model because it is cheap and fast
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def process_pdf_into_memory(pdf_path, class_id):
    """
    This function takes a PDF file and a Class ID (like 'Math-101').
    It chops the PDF into small pieces and saves them into the Pinecone database.
    """
    
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()
    
    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)
    
    # Connect to the Pinecone Database
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        namespace=class_id 
    )
    
    return f"Successfully processed {len(documents)} chunks for class {class_id}"

def ask_socratic_ai(question, class_id):
    """
    This function takes a student's question and the Class ID.
    It looks up relevant notes and generates a helpful, guiding answer.
    """
    
    # Connect to the database for THIS SPECIFIC class
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        namespace=class_id
    )
    
    # Create a Retriever that will look for the 3 most relevant notes
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Define the AI's Personality
    template = """
    You are a helpful and encouraging tutor for the class {class_id}.
    Your goal is to help the student understand the material, not just give them the answers.
    When asked a question about things that appear in the class notes, answer the question directly
    and quote the class notes in your answer. If the question is based on the class notes, and
    it sounds like a homework/test question, DO NOT give the direct answer but guide the student
    towards the answer by reminding them of relevant concepts from the notes.
    
    RULES:
    1. Use ONLY the provided Context (class notes) to answer.
    2. Do NOT give the direct answer to homework questions. Instead, ask a guiding question or give a hint.
    3. If the answer is not in the notes, say "I don't see that in our class notes."
    
    Context from Teacher's Notes:
    {context}
    
    Student's Question:
    {question}
    
    Your Response:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # Retrieve Notes -> Insert into Prompt -> Send to Model -> Read Answer
    chain = (
        {"context": retriever, "question": RunnablePassthrough(), "class_id": lambda x: class_id}
        | prompt
        | model
        | StrOutputParser()
    )

    # Run the chain and return the answer
    return chain.invoke(question)