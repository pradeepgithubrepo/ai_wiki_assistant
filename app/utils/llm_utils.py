import os
import langchain
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

class Helperclass:
    def __init__(self):
        pass

    def openai_client(self):
        client = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            model=os.getenv("AZURE_MODEL"),
            api_version=os.getenv("AZURE_API_VERSION"),
            api_key=os.getenv("AZURE_API_KEY"),
            openai_api_type="azure",
            temperature=0.0
        )
        return client

    def gemini_client(self):
        """
        Creates and returns a Google Gemini client using the API key from environment variables.
        """
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return embeddings_model

