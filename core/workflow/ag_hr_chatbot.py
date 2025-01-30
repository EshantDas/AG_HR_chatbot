from langchain_core.messages import HumanMessage, AIMessage
from core.pydantic_classes.workflow_clases import EvaluationResponse, Reframe_response
from core.prompts.workflow_prompts import eval_prompt, refine_reply_prompt
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI

load_dotenv()


llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2023-03-15-preview",
    temperature=0,
    seed=42,
)

llm_advance = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2023-03-15-preview",
    temperature=0.5,
    seed=42,
)


structured_llm = llm.with_structured_output(EvaluationResponse)
versative_structured_llm = llm_advance.with_structured_output(Reframe_response)


chain = eval_prompt | structured_llm
reframe_chain = refine_reply_prompt | versative_structured_llm


# async def get_concise_human_feedback(current_reply,question):


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
        reframed_reply = await reframe_chain.ainvoke(
            {"answer": answer, "response": response.reply}
        )
        return (
            response.score,
            reframed_reply.reframed_response,
            None,
            chat_history,
            chat_summary,
        )
    else:
        chat_summary.append({"Question": question, "Answer": answer})
        return (
            response.score,
            None,
            response.follow_up_question,
            chat_history,
            chat_summary,
        )
