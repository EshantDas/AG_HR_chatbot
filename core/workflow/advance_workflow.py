from core.prompts.advance_question_specific_prompts import question_chain_pair
from langchain_core.messages import HumanMessage, AIMessage


async def get_chain_for_question(
    question, dictionary=question_chain_pair
):  # -> Any | None:
    if question in dictionary:
        return dictionary[question]
    else:
        return None


async def return_reponse(question, answer, chat_history, chat_summary):
    chain = await get_chain_for_question(question)
    if chain:
        chat_history_first = chat_history[:3]
        response = await chain.ainvoke(
            {"chat_history": chat_history_first, "question": question, "answer": answer}
        )

        chat_history.append(AIMessage(content=question))
        chat_history.append(HumanMessage(content=answer))
        chat_history.append(AIMessage(content=response.reply))
        if response.score == 1:
            chat_summary.append({"Question": question, "Answer": answer})
            return response.score, None, None, chat_history, chat_summary
        elif response.score == 0:
            return (
                response.score,
                response.reply,
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
    else:
        return None
