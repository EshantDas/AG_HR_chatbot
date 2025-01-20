from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field


# Pydantic Class for Evaluation
class EvaluationResponseFollowup(BaseModel):
    """
    Evaluation response for verifying coherence, relevance, and alignment of user answers.
    """

    score: int = Field(
        ...,
        description="0 if the answer is not related to the question or is something not logical, 1 if the answer is relevant . It should be 1 if the answer is some information and not starting with yes but obviously it has to be relevant 2 if a follow-up question can be asked.",
    )
    reply: str = Field(
        ...,
        description="A polite response to guide the user if the score is 0, or an empty string if the score is 1.",
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="A follow-up question to ask if the score is 2, otherwise None.",
    )


# Pydantic Class for Evaluation
class EvaluationResponseOnly(BaseModel):
    """
    Evaluation response for verifying coherence, relevance, and alignment of user answers.
    """

    score: int = Field(
        ...,
        description="0 if the answer is not related to the question or is something not logical, 1 if the answer is relevant",
    )
    reply: str = Field(
        ...,
        description="A polite response to guide the user if the score is 0, or an empty string if the score is 1.",
    )
