import asyncio
import os
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
_CHAT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")

# Before implementation open the `vector_based_grounding.png` to see the flow of app

SYSTEM_PROMPT = """You are a RAG-powered assistant. The user message contains:
1. RAG CONTEXT - retrieved documents relevant to the user's question (from a similarity search).
2. USER QUESTION - the user's actual question.

Answer the user's question based ONLY on the provided RAG context. If the context does not contain relevant information, say so. Do not invent information."""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION:
{query}"""


def format_user_document(user: dict[str, Any]) -> str:
    lines = ["User:"]
    for key, value in user.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        user_client = UserClient()
        all_users = user_client.get_all_users()
        documents = [
            Document(page_content=format_user_document(user))
            for user in all_users
        ]
        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed; vectorstore is in-memory
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        batches = [
            documents[i : i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]
        stores = await asyncio.gather(
            *[FAISS.afrom_documents(batch, self.embeddings) for batch in batches]
        )
        if not stores:
            return FAISS.from_documents([], self.embeddings)
        final_vectorstore = stores[0]
        for store in stores[1:]:
            final_vectorstore.merge_from(store)
        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        docs_with_scores = await asyncio.to_thread(
            self.vectorstore.similarity_search_with_relevance_scores, query, k=k
        )
        context_parts = []
        for doc, relevance_score in docs_with_scores:
            if relevance_score >= score:
                context_parts.append(doc.page_content)
                print(f"score: {relevance_score}\n{doc.page_content}")
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        response = self.llm_client.invoke(messages)
        return response.content if isinstance(response.content, str) else ""


async def main():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        model="text-embedding-3-small-1",
        dimensions=384,
    )
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        azure_deployment=_CHAT_DEPLOYMENT,
        api_version=_CHAT_API_VERSION or "",
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = (await asyncio.to_thread(input, "> ")).strip()
            if user_question.lower() in ("quit", "exit"):
                break
            context = await rag.retrieve_context(user_question)
            augmented = rag.augment_prompt(user_question, context)
            answer = rag.generate_answer(augmented)
            print(answer)


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce