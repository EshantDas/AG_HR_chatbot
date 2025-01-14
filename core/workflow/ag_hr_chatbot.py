from langchain_core.messages import HumanMessage, AIMessage
from core.pydantic_classes.workflow_clases import EvaluationResponse
from core.prompts.workflow_prompts import eval_prompt
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

load_dotenv()


api_key: str | None = os.environ.get("OPENAI_API_KEY2")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
structured_llm = llm.with_structured_output(EvaluationResponse)

chain = eval_prompt | structured_llm


async def ag_hr_bot_response(
    question, answer, chat_history, chat_summary, follow_up=False
):
    response = await chain.ainvoke(
        {"chat_history": chat_history, "question": question, "answer": answer}
    )
    chat_history.append(AIMessage(content=question))
    chat_history.append(HumanMessage(content=answer))
    chat_history.append(AIMessage(content=response.reply))
    if response.score == 1:
        chat_summary.append({"Question": question, "Answer": answer})
        return response.score, None, None, chat_history, chat_summary
    elif response.score == 0:
        return response.score, response.reply, None, chat_history, chat_summary
    else:
        chat_summary.append({"Question": question, "Answer": answer})
        return (
            response.score,
            None,
            response.follow_up_question,
            chat_history,
            chat_summary,
        )
