import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        print("Initializing Microwave Manual RAG System...")
    
        if os.path.exists("microwave_faiss_index"):
            print("Loading FAISS vectorstore from local index with folder_path 'microwave_faiss_index' and configured embeddings, allowing dangerous deserialization.")
            return FAISS.load_local(
                folder_path="microwave_faiss_index",
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new index as 'microwave_faiss_index' folder does not exist.")
            return self._create_new_index()

    def _create_new_index(self) -> VectorStore:
        print("Loading text document from 'microwave_manual.txt' with 'utf-8' encoding.")
        loader = TextLoader(file_path="microwave_manual.txt", encoding="utf-8")
    
        print("Loading documents using the loader.")
        documents = loader.load()
    
        print("Creating RecursiveCharacterTextSplitter with chunk_size=300, chunk_overlap=50, and separators.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."]
        )
    
        print("Splitting documents into chunks.")
        chunks = text_splitter.split_documents(documents)
    
        print("Creating vectorstore from document chunks.")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
    
        print("Saving indexed data locally to 'microwave_faiss_index'.")
        vectorstore.save_local("microwave_faiss_index")
    
        print("Returning the created vectorstore.")
        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")
        
        print("Performing similarity search with relevance scores on the vectorstore.")
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score
        )
        
        context_parts = []
        print("Iterating through retrieved results:")
        for doc, score in results:
            context_parts.append(doc.page_content)
            print(f"Score: {score}")
            print(f"Content: {doc.page_content}")
        
        print("=" * 100)
        return "\n\n".join(context_parts) # will join all chunks ion one string with `\n\n` separator between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")
    
        print("Formatting USER_PROMPT template with context and query.")
        augmented_prompt = USER_PROMPT.format(context=context, query=query)
    
        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")
    
        print("Creating messages array with SystemMessage and HumanMessage.")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
    
        print("Invoking LLM client with messages.")
        response = self.llm_client.invoke(messages)
    
        print(f"Response: {response.content}")
        return response.content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
    
        print("Executing STEP 1: Retrieval of context.")
        context = rag.retrieve_context(user_question)
    
        print("Executing STEP 2: Augmentation.")
        augmented_prompt = rag.augment_prompt(user_question, context)
    
        print("Executing STEP 3: Generation.")
        answer = rag.generate_answer(augmented_prompt)



main(
    MicrowaveRAG(
        embeddings=AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY)
        ),
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version=""
        )
    )
)