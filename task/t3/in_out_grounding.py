import asyncio
import json
import os
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, RootModel
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
_CHAT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")

# Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)

NEE_SYSTEM_PROMPT = """You are a Named Entity Extraction system for a hobbies search. Given a user query and a list of user profiles (id and about_me only), extract the hobbies or interests that match the query and list the user IDs for each hobby.

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{"hobby_name": [user_id1, user_id2, ...], "another_hobby": [user_id3, ...]}}

Use the exact hobby/interest terms that appear in the user profiles. User IDs must be integers from the provided profiles only."""

NEE_USER_PROMPT = """## User query:
{query}

## User profiles (id and about_me):
{context}

Return JSON mapping hobby names to lists of user IDs."""


class HobbyGroups(RootModel[dict[str, list[int]]]):
    """Mapping of hobby name to list of user IDs from NEE."""


def _doc_content(user: dict[str, Any]) -> str:
    """Build page_content from user id and about_me only."""
    return f"id: {user.get('id', '')}\nabout_me: {user.get('about_me', '')}"


def _users_to_documents(users: list[dict[str, Any]]) -> list[Document]:
    return [
        Document(
            id=str(u["id"]),
            page_content=_doc_content(u),
            metadata={"user_id": u["id"]},
        )
        for u in users
    ]


async def _sync_vectorstore(
    vectorstore: Chroma,
    user_client: UserClient,
) -> None:
    """Update vectorstore: remove deleted users, add new ones."""
    current_users = user_client.get_all_users()
    current_ids = {str(u["id"]) for u in current_users}

    try:
        existing = vectorstore._collection.get(include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    ids_to_remove = existing_ids - current_ids
    if ids_to_remove:
        vectorstore.delete(ids=list(ids_to_remove))

    ids_to_add = current_ids - existing_ids
    if ids_to_add:
        add_users = [u for u in current_users if str(u["id"]) in ids_to_add]
        new_docs = _users_to_documents(add_users)
        await vectorstore.aadd_documents(new_docs)


async def _output_grounding(
    user_client: UserClient,
    hobby_groups: dict[str, list[int]],
) -> dict[str, list[dict[str, Any]]]:
    """Fetch full user info for each ID and return hobby -> list of user dicts."""
    result: dict[str, list[dict[str, Any]]] = {}

    for hobby, ids in hobby_groups.items():
        tasks = [user_client.get_user(uid) for uid in ids]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        users = []
        for r in responses:
            if isinstance(r, Exception):
                continue
            users.append(r)
        result[hobby] = users

    return result


async def main() -> None:
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        model="text-embedding-3-small-1",
        dimensions=384,
    )
    llm = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        azure_deployment=_CHAT_DEPLOYMENT,
        api_version=_CHAT_API_VERSION or "",
    )
    user_client = UserClient()

    # In-memory Chroma collection
    vectorstore = Chroma(
        collection_name="hobbies",
        embedding_function=embeddings,
    )

    print("Cold start: loading users into vectorstore...")
    all_users = user_client.get_all_users()
    if all_users:
        docs = _users_to_documents(all_users)
        await vectorstore.aadd_documents(docs)
    print("Ready. Query samples: 'I need people who love to go to mountains'")

    parser = PydanticOutputParser(pydantic_object=HobbyGroups)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NEE_SYSTEM_PROMPT),
            ("human", NEE_USER_PROMPT),
        ]
    )

    while True:
        user_question = (await asyncio.to_thread(input, "> ")).strip()
        if user_question.lower() in ("quit", "exit"):
            break

        # Per-request sync: remove deleted, add new users
        await _sync_vectorstore(vectorstore, user_client)

        # Retrieve relevant docs (id + about_me)
        docs = await asyncio.to_thread(
            vectorstore.similarity_search,
            user_question,
            k=50,
        )
        context = "\n\n".join(doc.page_content for doc in docs)

        if not context.strip():
            print("No relevant users found.")
            continue

        chain = prompt | llm | parser
        try:
            parsed = chain.invoke({"query": user_question, "context": context})
            hobby_mapping = parsed.root if isinstance(parsed, HobbyGroups) else {}
        except Exception as e:
            print(f"NEE parse error: {e}")
            continue

        if not hobby_mapping:
            print("No hobbies extracted.")
            continue

        # Output grounding: fetch full user info for each ID
        grounded = await _output_grounding(user_client, hobby_mapping)
        print(json.dumps(grounded, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
