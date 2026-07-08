import warnings
from google.adk.agents import Agent
from google.adk.agents.sequential_agent import SequentialAgent
import pandas as pd
from pydantic import BaseModel, Field
import json
import random

# The ADK team has deprecated SequentialAgent in favor of `Workflow`, but as of
# this ADK version `Workflow` cannot yet be used as a sub-agent of an LlmAgent
# (see https://github.com/google/adk-python/discussions/5581). Since we need
# data_query_agent to be a sub-agent of root_agent, SequentialAgent is still
# the only supported option — silence just this warning until ADK adds support.
warnings.filterwarnings(
    "ignore",
    message="SequentialAgent is deprecated.*",
    category=DeprecationWarning,
)

MODEL = "gemini-2.0-flash"  # stable id instead of the experimental -exp variant

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

brands = ["Dior", "Chanel", "Creed", "Tom Ford", "YSL", "Gucci", "Versace", "Armani", "Calvin Klein", "Burberry"]
concentrations = ["Eau de Toilette", "Eau de Parfum", "Parfum", "Cologne"]
genders = ["Men", "Women", "Unisex"]
availability_status = ["In Stock", "Limited Stock", "Out of Stock"]
fragrance_families = ["Woody", "Floral", "Oriental", "Fresh", "Fruity", "Citrus", "Spicy"]
seasons = ["Summer", "Winter", "Spring", "Fall"]
launch_years = list(range(2000, 2024))
adjectives = ["Mystic", "Velvet", "Golden", "Noir", "Crystal", "Amber", "Silken", "Wild", "Intense", "Fresh"]
scent_keywords = ["Whisper", "Flame", "Dream", "Aura", "Pulse", "Echo", "Bloom", "Rush", "Mist", "Twilight"]


def generate_unique_perfume_data(n=30):
    used_names = set()
    products = []

    while len(products) < n:
        brand = random.choice(brands)
        name = f"{random.choice(adjectives)}{random.choice(scent_keywords)} by {brand}"
        if name in used_names:
            continue
        used_names.add(name)
        product = {
            "Product Name": name,
            "Brand": brand,
            "Top Notes": ", ".join(random.sample(["Bergamot", "Lemon", "Mandarin", "Apple", "Pear"], 2)),
            "Heart Notes": ", ".join(random.sample(["Jasmine", "Rose", "Lavender", "Cinnamon", "Cardamom"], 2)),
            "Base Notes": ", ".join(random.sample(["Musk", "Amber", "Cedarwood", "Patchouli", "Vanilla"], 2)),
            "Concentration": random.choice(concentrations),
            "Gender": random.choice(genders),
            "Price (USD)": round(random.uniform(50, 400), 2),
            "Availability": random.choice(availability_status),
            "Fragrance Family": random.choice(fragrance_families),
            "Best Season": random.choice(seasons),
            "Launch Year": random.choice(launch_years),
            "Rating (out of 5)": round(random.uniform(3.0, 5.0), 1),
        }
        products.append(product)

    return products


df = pd.DataFrame(generate_unique_perfume_data(30))

# ---------------------------------------------------------------------------
# Agent 1: refine the user's natural language query
# ---------------------------------------------------------------------------

query_refiner_agent = Agent(
    name="query_refiner_agent_v1",
    model=MODEL,
    description="Refines the user's query to be more specific and accurate",
    instruction=(
        "You are a helpful assistant that refines user queries about perfumes to be more specific and accurate. "
        "If the user asks for a best seller or a random suggestion, rewrite the query so it asks for the "
        "perfume(s) with the maximum rating. "
        "Respond with ONLY the refined query in plain English — no instructions, no explanations, no extra text."
    ),
    output_key="refined_query",  # stored in session.state["refined_query"] for the next agent
)


class QueryGeneratorOutput(BaseModel):
    query: str = Field(description="A valid pandas expression that can be used to query the data.")


# ---------------------------------------------------------------------------
# Agent 2: turn the refined query into a pandas expression
# ---------------------------------------------------------------------------

query_generator_agent = Agent(
    name="query_generator_agent_v1",
    model=MODEL,
    description="Turns a refined natural-language query into a pandas expression",
    instruction=(
        "You are a helpful assistant that writes pandas expressions against a DataFrame called `df`.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Sample rows:\n{df.head().to_string()}\n\n"
        "The refined user request is:\n{refined_query}\n\n"
        "Respond ONLY with a JSON object matching the required schema, containing a single valid "
        "pandas expression (operating on `df`) that answers the request. "
        "Only use read-only operations (filtering, sorting, selecting columns) — never assignments, "
        "imports, or function calls outside of pandas/df methods."
    ),
    output_schema=QueryGeneratorOutput,
    output_key="generated_query",  # stores the JSON string in session.state["generated_query"]
)


# ---------------------------------------------------------------------------
# Tool + Agent 3: execute the pandas expression safely and summarize results
# ---------------------------------------------------------------------------

_FORBIDDEN_TOKENS = ("import", "__", "os.", "sys.", "open(", "exec(", "eval(", "subprocess")


def execute_query(query: str) -> dict:
    """Executes a pandas expression against the perfume DataFrame and returns JSON-serializable results."""
    lowered = query.lower()
    if any(tok in lowered for tok in _FORBIDDEN_TOKENS):
        return {"error": "Query rejected: contains disallowed operations."}

    try:
        result = eval(query, {"__builtins__": {}}, {"df": df, "pd": pd})

        if isinstance(result, pd.Series):
            return {"results": result.to_dict()}
        if isinstance(result, pd.DataFrame):
            return {"results": result.to_dict(orient="records")}

        if result is None:
            return {"error": "No products found matching your criteria"}
        return {"results": result}

    except Exception as e:
        return {"error": f"Query failed: {e}"}


query_execution_agent = Agent(
    name="query_execution_agent_v1",
    model=MODEL,
    description="Executes the generated pandas query and returns a human-readable summary",
    instruction=(
        "You will receive a JSON object in {generated_query} containing a `query` field with a pandas "
        "expression. Call the `execute_query` tool with that expression, then present the results to the "
        "user in a concise, friendly, human-readable way. If there's an error or no results, say so clearly "
        "and suggest the user try a different request."
    ),
    tools=[execute_query],
)

data_query_agent = SequentialAgent(
    name="data_query_agent_v1",
    description=(
        "Orchestrates data query processing by sequentially invoking "
        "query_refiner_agent_v1 -> query_generator_agent_v1 -> query_execution_agent_v1"
    ),
    sub_agents=[query_refiner_agent, query_generator_agent, query_execution_agent],
)

# ---------------------------------------------------------------------------
# Root agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="general_agent_v1",
    model=MODEL,
    description="Handles general conversation, represents the Scentara brand, and delegates data queries to `data_query_agent`",
    instruction=(
        "You are a friendly and talkative assistant representing Scentara, a premium perfume brand known for "
        "its exquisite fragrances. Introduce yourself as Scentara's brand representative and share our passion "
        "for creating unique, high-quality perfumes that captivate the senses. Engage users warmly, answer "
        "general questions, and encourage them to ask about our perfumes, collections, or anything related to "
        "the Scentara brand. "
        "If the user asks about perfumes or products, delegate the query to `data_query_agent`. "
        "Always end your responses by asking an engaging question to keep the conversation going, such as "
        "'What kind of fragrance are you looking for?' or 'Have you tried any of our signature scents yet?'"
    ),
    sub_agents=[data_query_agent],
)