import os

from dotenv import load_dotenv
from groq import Groq, APIError


# ============================================================
# CONFIGURATION
# ============================================================

# Load API credentials and model configuration from .env
load_dotenv()

MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


def get_client():
    """Initialize the Groq API client."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Create a .env file in the project root "
            "and add your Groq API key."
        )

    return Groq(api_key=api_key)


# ============================================================
# SHARED AGENT EXECUTION FUNCTION
# ============================================================

def run_agent(agent_name, instructions, user_data):
    """
    Execute a specialized AI agent using the Groq API.

    Args:
        agent_name: Name of the agent.
        instructions: Agent-specific system instructions.
        user_data: Information provided to the agent.

    Returns:
        The agent's generated response as a string.
    """

    try:
        client = get_client()

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": instructions
                },
                {
                    "role": "user",
                    "content": user_data
                }
            ],
            temperature=0.2,
            max_tokens=800
        )

        result = response.choices[0].message.content

        if not result or not result.strip():
            raise RuntimeError(
                f"{agent_name} returned an empty response."
            )

        return result.strip()

    except APIError as error:
        raise RuntimeError(
            f"{agent_name} encountered an AI API error. "
            "Check your API key, model availability, "
            "network connection, and API quota."
        ) from error

    except Exception as error:
        if isinstance(error, (ValueError, RuntimeError)):
            raise

        raise RuntimeError(
            f"{agent_name} could not complete its task: {error}"
        ) from error


# ============================================================
# AGENT 1: SKIN ANALYSIS AGENT
# ============================================================

def skin_analysis_agent(skin_type, concerns, sensitivity):
    """
    Interpret the user's stated skin profile and priorities.
    """

    instructions = """
    You are the Skin Analysis Agent of Derma Tracker AI.

    YOUR RESPONSIBILITIES:
    1. Interpret the user's stated skin type.
    2. Analyze their reported skincare concerns.
    3. Consider their reported skin sensitivity.
    4. Identify the three most relevant skincare priorities.
    5. Explain what additional information would be useful.

    OUTPUT FORMAT:
    Use these headings:

    1. Skin Profile
    2. Main Concerns
    3. Recommended Priorities
    4. Information Gaps

    RULES:
    - Use only the information supplied by the user.
    - Do not diagnose skin diseases.
    - Do not claim certainty from limited information.
    - Do not infer a medical condition from a skin type.
    - Do not prescribe medication.
    - Explain uncertainty when information is missing.
    - Keep recommendations practical and understandable.
    """

    user_data = f"""
    USER SKIN PROFILE

    Skin type: {skin_type}
    Skin concerns: {", ".join(concerns)}
    Skin sensitivity: {sensitivity}

    Interpret this profile and identify appropriate skincare
    priorities without diagnosing a medical condition.
    """

    return run_agent(
        "Skin Analysis Agent",
        instructions,
        user_data
    )


# ============================================================
# AGENT 2: INGREDIENT INTELLIGENCE AGENT
# ============================================================

def ingredient_intelligence_agent(ingredients, allergies):
    """
    Explain supplied skincare ingredients and potential cautions.
    """

    instructions = """
    You are the Ingredient Intelligence Agent of Derma Tracker AI.

    YOUR RESPONSIBILITIES:
    1. Explain the general functions of the supplied ingredients.
    2. Describe their commonly recognized skincare benefits.
    3. Identify potential irritation risks.
    4. Consider reported allergies and previous reactions.
    5. Explain possible compatibility concerns.
    6. Identify information that should be verified.

    OUTPUT FORMAT:
    Use these headings:

    1. Ingredient Overview
    2. Potential Benefits
    3. Potential Risks
    4. Compatibility Notes
    5. What to Verify

    RULES:
    - Analyze only ingredients actually supplied.
    - Do not invent product ingredient lists.
    - A product name alone is not enough to establish its
      complete ingredient composition.
    - Do not assume every ingredient combination is incompatible.
    - Consider concentration, formulation, frequency of use,
      and individual tolerance when relevant.
    - Do not prescribe medication or diagnose diseases.
    - If information is insufficient, state that clearly.
    - If an allergy is reported, flag the uncertainty and advise
      avoiding the suspected trigger until its safety is clarified.
    """

    user_data = f"""
    USER INGREDIENT INFORMATION

    Products or ingredients:
    {ingredients}

    Reported allergies or reactions:
    {allergies}

    Review the supplied information and explain potential
    benefits, limitations, and safety considerations.
    """

    return run_agent(
        "Ingredient Intelligence Agent",
        instructions,
        user_data
    )


# ============================================================
# AGENT 3: SKINCARE ROUTINE AGENT
# ============================================================

def skincare_routine_agent(
    skin_type,
    concerns,
    budget,
    analysis_result,
    ingredient_result
):
    """
    Integrate the first two agents' outputs into a skincare routine.
    """

    instructions = """
    You are the Skincare Routine Agent of Derma Tracker AI.

    You receive the user's skin profile and the results produced
    by the Skin Analysis and Ingredient Intelligence agents.

    YOUR RESPONSIBILITIES:
    1. Review the skin analysis.
    2. Review ingredient-related cautions.
    3. Generate a simple morning skincare routine.
    4. Generate a simple evening skincare routine.
    5. Respect the user's budget.
    6. Avoid unnecessary products and excessive active ingredients.
    7. Explain how to introduce new products cautiously.

    OUTPUT FORMAT:
    Use these headings:

    1. Morning Routine
    2. Evening Routine
    3. Introduction Plan
    4. Safety Precautions

    RULES:
    - Use the earlier agents' outputs as context, not as
      unquestionable medical facts.
    - Prioritize a gentle cleanser when needed, moisturizer,
      and broad-spectrum sunscreen during the day.
    - Avoid introducing several new active ingredients at once.
    - Consider reported allergies and unresolved safety concerns.
    - Do not recommend a suspected allergen.
    - If allergy safety is uncertain, recommend verifying
      product ingredients before use.
    - Avoid adding strong active ingredients when sensitivity
      or current product use is unclear.
    - Do not prescribe medication or diagnose diseases.
    - Do not promise results within a specific number of days.
    - Do not invent product prices or ingredient compositions.
    - State assumptions when information is insufficient.
    - Recommend a qualified clinician for persistent, severe,
      painful, or worsening skin problems.
    """

    user_data = f"""
    USER PROFILE

    Skin type: {skin_type}
    Skin concerns: {", ".join(concerns)}
    Monthly budget: {budget}

    OUTPUT FROM AGENT 1: SKIN ANALYSIS

    {analysis_result}

    OUTPUT FROM AGENT 2: INGREDIENT INTELLIGENCE

    {ingredient_result}

    TASK

    Integrate the information above into a practical, conservative
    morning and evening skincare routine.

    Explain the purpose of each step and identify relevant
    precautions. Do not diagnose or prescribe medication.
    """

    return run_agent(
        "Skincare Routine Agent",
        instructions,
        user_data
    )


# ============================================================
# ORCHESTRATOR: COORDINATE ALL THREE AGENTS
# ============================================================

def run_skincare_workflow(
    skin_type,
    concerns,
    sensitivity,
    ingredients,
    allergies,
    budget
):
    """
    Execute all three agents sequentially.

    Agent 1 -> Agent 2 -> Agent 3

    Agent 3 receives the outputs of Agents 1 and 2.
    If an agent fails, the workflow stops and reports the error.
    """

    # Validate required inputs before making API calls.
    if not skin_type:
        raise ValueError("Please provide a skin type.")

    if not concerns:
        raise ValueError("Please select at least one skin concern.")

    # AGENT 1
    analysis_result = skin_analysis_agent(
        skin_type,
        concerns,
        sensitivity
    )

    # AGENT 2
    ingredient_result = ingredient_intelligence_agent(
        ingredients or "No ingredients provided.",
        allergies or "Not specified."
    )

    # AGENT 3
    routine_result = skincare_routine_agent(
        skin_type,
        concerns,
        budget,
        analysis_result,
        ingredient_result
    )

    # Return individual outputs so the UI can display each result.
    return {
        "skin_analysis": analysis_result,
        "ingredient_intelligence": ingredient_result,
        "skincare_routine": routine_result
    }

