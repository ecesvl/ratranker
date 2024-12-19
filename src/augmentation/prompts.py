from typing import List

from langchain_core.prompts import PromptTemplate
from loguru import logger

from globals import (EXPLANATION, FACTUALITY, INFORMATION_DENSITY,
                     COMMONSENSE, TEXTUAL_DESCRIPTION, ALL_RATIONALES_LIST)

from examples import list_examples_dict

# Data augmentation prompt templates to construct own prompts,
# allowing to include and exclude rationales.

# constant prompt parts
BASE_PROMPT = """
In this task you will be presented with a query, a document passage and a label indicating if the document is relevant 
for answering the query. Your objective is to create rationales that explain why this document passage is relevant or not.
You should answer following rationales:
"""

ANSWER_FORMAT_PROMPT = """
{format_instruction}
"""

TASK_PROMPT = """
Please read the query, passage and label carefully, your answers should coincide with the label whether it is relevant or not:
    # Query: {query}
    # Passage: {passage_text}
    # Label: {label}
"""

FEW_SHOT_EXAMPLES_PROMPT = """
Here are some examples.
"""

# dicts containing rationales, which can be added to prompt
rationales_parts = {
    EXPLANATION: "explaining why the document is either relevant or not",
    FACTUALITY: "an assessment about truthfulness",
    INFORMATION_DENSITY: "an assessment about information density (category: Low/Moderate/High)",
    COMMONSENSE: "a brief evaluation of the document's alignment with general world knowledge and practical reasoning",
    TEXTUAL_DESCRIPTION: "keywords about style and structure of the document"
}


def create_prompt(rationales: List[str], few_shot: bool, format_instruction: str, num_examples: int = None) -> PromptTemplate:
    logger.debug("Creating prompt template..")
    rational_prompt = create_rationales_prompt(rationales)
    examples_prompts = FEW_SHOT_EXAMPLES_PROMPT + "\n".join(
        create_examples_prompts(rationales=rationales, num_examples=num_examples)) if few_shot else ""

    complete_prompt_str = rational_prompt + ANSWER_FORMAT_PROMPT + examples_prompts + TASK_PROMPT
    partial_prompt = PromptTemplate.from_template(complete_prompt_str,
                                                  partial_variables={"format_instruction": format_instruction})
    return partial_prompt


def create_rationales_prompt(rationales: List[str]) -> str:
    assert set(rationales).issubset(ALL_RATIONALES_LIST), "Only rationales from all_rationales list are supported"
    r_prompt = [f"- {rationale}: {rationales_parts.get(rationale)}" for rationale in rationales]
    return BASE_PROMPT + "    " + "\n    ".join(r_prompt) + "\n"


def create_examples_prompts(rationales: List[str], num_examples):
    # List of prompt parts based on the augmentation configuration

    examples = list_examples_dict[:num_examples]

    list_example_prompts = []
    for example in examples:
        prompt_parts =[]
        for rationale in rationales:
            prompt_parts.append(f'"{rationale}": "{example[rationale]}')

        # Filter out None values and join the parts with commas and newlines
        prompt_parts_filtered = filter(None, prompt_parts)
        prompt_string = ",\n        ".join(prompt_parts_filtered)

        # Construct the final prompt template
        prompt_template = f"""
        Example {list_examples_dict.index(example)}:
        # Query: "{example['query']}"
        # Passage: "{example['passage']}"
        # Label: {example['label']}
        
        {{{{
        {prompt_string}
        }}}}
        """.strip()  # Strip to remove any leading/trailing whitespace
        list_example_prompts.append(prompt_template)

    return list_example_prompts


