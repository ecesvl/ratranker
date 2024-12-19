from enum import Enum
from typing import Optional, List

import pandas as pd
from loguru import logger
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from prompts import create_prompt
from src.utils.llm import get_gpt4o, get_gpt4turbo
from globals import ALL_RATIONALES_LIST, GPT_CONFIG

__all__ = ['Augmenter', 'AugmentConfig']

def create_batches(dataframe, batch_size):
    # This will hold the list of batches
    batches = []

    # Calculate the number of batches
    num_batches = len(dataframe) // batch_size + (0 if len(dataframe) % batch_size == 0 else 1)

    for i in range(num_batches):
        # Calculate start and end indices for the current batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        # Slice the dataframe to create a new batch
        batch = dataframe.iloc[start_idx:end_idx]
        batches.append(batch)

    return batches

class Density(Enum):
    High = "High"
    Moderate = "Moderate"
    Low = "Low"


class Rationales(BaseModel):
    explanation: Optional[str] = None
    factuality: Optional[str] = None
    information_density: Optional[Density] = None
    commonsense: Optional[str] = None
    textual_description: Optional[str] = None

class AugmentConfig(BaseModel):
    rationales: Optional[List[str]] = ALL_RATIONALES_LIST
    few_shot: bool = True
    num_examples: Optional[int] = 3


class Augmenter:
    def __init__(self, aug_config: AugmentConfig, gpt4_config: str = GPT_CONFIG):
        logger.info(f"Using {gpt4_config} for data augmentation")
        self.parser = PydanticOutputParser(pydantic_object=Rationales)
        self.format_instructions = self.parser.get_format_instructions()
        if gpt4_config == 'gpt4-turbo':
            self.llm = get_gpt4turbo().bind(response_format={"type": "json_object"})
        elif gpt4_config == 'gpt4o':
            self.llm = get_gpt4o().bind(response_format={"type": "json_object"})
        else:
            raise ValueError("Invalid gpt4_config value. Please choose 'gpt4-turbo' or 'gpt4o'.")
        self.prompt = create_prompt(rationales=aug_config.rationales, few_shot=aug_config.few_shot,
                                    num_examples=aug_config.num_examples, format_instruction=self.format_instructions)
        self.chain = self.prompt | self.llm | self.parser

    def augment(self, query: str, passage: str, label: str):
        logger.debug(f'Creating rationales for query: "{query}"')
        input_var = {"query": query, "passage_text": passage, "label": label}
        return self.chain.invoke(input_var)

    def augment_multiple(self, queries: List[str], passages: List[str], labels: List[str]):
        assert (len(queries) == len(passages) == len(labels))
        logger.debug(f'Creating rationales for multiple queries {queries}')
        input_vars = []
        for query, passage, label in zip(queries, passages, labels):
            input_var = {"query": query, "passage_text": passage, "label": label}
            input_vars.append(input_var)
        return self.chain.batch(input_vars)

    def process_output(self):
        # TODO
        pass

    def augment_batches(self, dataframe: pd.DataFrame, batch_size: int):
        batches = create_batches(dataframe, batch_size)
        # This will hold the augmented datasets

        for batch in batches:
            # Extract queries, passages, and labels from the current batch
            queries = batch['query'].tolist()
            passages = batch['passage'].tolist()
            labels = batch['label'].tolist()

            # Use the augment_multiple method
            augmented_batch = self.augment_multiple(queries, passages, labels)

            # Collect the results
            yield augmented_batch



