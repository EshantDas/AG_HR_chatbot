from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field


# Pydantic Class for Evaluation
class EvaluationResponse(BaseModel):
    """
    Evaluation response for verifying coherence, relevance, and alignment of user answers.
    """

    score: int = Field(
        ...,
        description="0 if the answer is not related to the question, 1 if the answer is relevant, 2 if a follow-up question can be asked.",
    )
    reply: str = Field(
        ...,
        description="A polite response to guide the user if the score is 0, or an empty string if the score is 1.",
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="A follow-up question to ask if the score is 2, otherwise None.",
    )


class Reframe_response(BaseModel):
    """
    Represents a refined, concise, and friendly rephrased response to a given answer.

    Attributes:
    reframed_response (str): The rephrased, human-like response that is clear and to the point.
    """

    reframed_response: str = Field(
        description="The revised, friendly, and concise response after rephrasing the original answer."
    )
