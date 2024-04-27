from enum import Enum


class PromptingTechnique(Enum):
    CHAIN_OF_THOUGHT: str = "Chain-of-Thought"
    FEW_SHOT: str = "Few-Shot"
    ZERO_SHOT: str = "Zero-Shot"
    PROMPT_CHAINING: str = "Prompt Chaining"
