chat_history = []
chat_summary = []

for question in questions_list[:5]:
    answer = input(question)
    score, reply, follow_up, chat_history, chat_summary = await ag_hr_bot_response(
        question, answer, chat_history, chat_summary
    )
    if score == 0:
        while score == 0:  # Keep retrying while the score is 0
            (
                score,
                reply,
                follow_up,
                chat_history,
                chat_summary,
            ) = await ag_hr_bot_response(question, answer, chat_history, chat_summary)

    if score == 1:
        pass  # Go to the next question

    elif score == 2:
        while follow_up:  # Continue handling follow-ups until resolved
            follow_up_answer = input(follow_up)
            (
                score,
                reply,
                follow_up,
                chat_history,
                chat_summary,
            ) = await ag_hr_bot_response(
                follow_up, follow_up_answer, chat_history, chat_summary
            )
            if not follow_up:  # Exit the follow-up loop if no more follow-up questions
                break

        # Once all follow-ups are resolved, move to the next question
