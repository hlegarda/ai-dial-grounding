import asyncio
import os
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from openai import APIError, APIStatusError
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# Optional env: AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_API_VERSION (e.g. if you get "No route")
_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
_CHAT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")

# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }

llm_client = AzureChatOpenAI(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    azure_deployment=_CHAT_DEPLOYMENT,
    api_version=_CHAT_API_VERSION or "",
)
token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    parts = []
    for user in context:
        lines = ["User:"]
        for key, value in user.items():
            lines.append(f"  {key}: {value}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


async def generate_response(system_prompt: str, user_message: str, *, quiet: bool = False) -> str:
    if not quiet:
        print("Processing...")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    try:
        response = await llm_client.ainvoke(messages)
    except (APIError, APIStatusError) as e:
        print(
            f"LLM request failed: {e}. "
            "Check DIAL_API_KEY is set, you are on VPN if required, and the DIAL endpoint is available."
        )
        raise
    usage = response.response_metadata.get("token_usage", {})
    total_tokens = usage.get("total_tokens", 0)
    token_tracker.add_tokens(total_tokens)
    content = response.content if isinstance(response.content, str) else ""
    if not quiet:
        print(content)
        print(f"total_tokens: {total_tokens}")
    return content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = (await asyncio.to_thread(input, "> ")).strip()
    if user_question:
        print("\n--- Searching user database ---")

        user_client = UserClient()
        all_users = user_client.get_all_users()

        batch_size = 100
        user_batches = [
            all_users[i : i + batch_size]
            for i in range(0, len(all_users), batch_size)
        ]

        num_batches = len(user_batches)
        print(f"Processing {num_batches} batches in parallel...")
        tasks = [
            generate_response(
                BATCH_SYSTEM_PROMPT,
                USER_PROMPT.format(context=join_context(batch), query=user_question),
                quiet=True,
            )
            for batch in user_batches
        ]
        results = await asyncio.gather(*tasks)

        filtered = [r for r in results if r.strip() != "NO_MATCHES_FOUND"]

        if filtered:
            combined = "\n\n".join(filtered)
            await generate_response(
                FINAL_SYSTEM_PROMPT,
                USER_PROMPT.format(context=combined, query=user_question),
            )
        else:
            print("No users found matching")

        print("Token usage summary:", token_tracker.get_summary())


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation