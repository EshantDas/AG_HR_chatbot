from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from core.pydantic_classes.advance_workflow_classes import (
    EvaluationResponseFollowup,
    EvaluationResponseOnly,
)
from core.workflow.static_questions import questions_list

load_dotenv()


llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2023-03-15-preview",
    temperature=0,
    seed=42,
)

chains = []
llm_follow_up = llm.with_structured_output(EvaluationResponseFollowup)
llm_reply = llm.with_structured_output(EvaluationResponseOnly)


eval_system_prompt = """ 
You are an expert, agentic chatbot designed to evaluate user responses for verifying questions related to what to put in Job Descriptions related questions. Your tasks include verifying coherence, relevance, and alignment of user answers to the given questions.
You provide constructive, polite feedback with examples when answers are irrelevant or nonsensical, and ensure clarity in all interactions. Make sure you behave as a human rather than like a chatbot
Please move forward with simple spelling mistakes and dont be too strict on the spelling and just give score 1 if it user means something from the question but there is spelling mistakes.
When user ask suggestion or examples dont tell him his answer is not relevant . Just give him the suggestions or examples
If user asks some suggestions then be matured enough to provide them suggestions and make score 0 because you need to provide examples or suggestions
"""


prompt1_ = """ 
So the question is  "Let's begin with the basics. What's the job title you're hiring for?"
and the answer given by the user is "{answer}"


It should be for a single person only like a single position for example Product Ownder it cannot be Product Owners, It has to be Data Enineer it cannot be Data Engineers
Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be a specific Job title only. It cant be illogical or irrelevant
Additional Key Points 
1. It cannot be some jubbarish data.
3. Verify if the answer is coherent and not gibberish or nonsense.
4. Ensure the answer is relevant to the question and not completely invalid or unrelated.
Also check the edge case examples I am giving you handle them accordingly

On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevant, Now if the score is 0 then you need to reply very carefully and very logically.


You
Important Points while replying:
- Carefully understand what went wrong on the answer based on the question and check if the answer is not logical or jubbarish not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

Now this needs to be 100 percent correct answer and it should not be any margin of error .
<Edge Case Examples>
For example if some one answers Solution Architect give a polite reply maybe he meant Solutions Architect because Solutions Architect is more general term rather Solution Architect so give a reply like that. So the score would still be 0 and give a reply like that
There can be other Examples like Chief Operations Officer and if user answer Chief Operation Officer it will be wrong 
Correct Usage: Plural Forms Preferred
Chief Operations Officer

Why? "Operations" refers to the broad scope of activities managed, not a single "operation."
Solutions Architect

Why? It emphasizes creating multiple solutions across systems or problems, which is the role's nature.
Facilities Manager

Why? Refers to managing multiple facilities rather than a single facility.
Customer Success Manager

Why? The focus is on ensuring the success of multiple customers, not just one.
Data Analytics Specialist

Why? Focuses on analyzing a variety of datasets, not just a single analysis.
Corporate Affairs Manager

Why? Involves handling a wide range of corporate affairs, not just one affair.
There can be may more examples like this so think dynamically becuase for this case there cannot be even a single letter mismatch 


Now there are some correct Usage of Singular Forms Preferred
Correct Usage: Singular Forms Preferred
Chief Financial Officer (CFO)

Why? Refers to overseeing the singular domain of "finance" as a collective entity.
Chief Technology Officer (CTO)

Why? Refers to overseeing the singular domain of "technology," not multiple "technologies."
Human Resources Manager

While "resources" is plural, this is a collective singular term for the HR function, not multiple individual resources.
Product Manager

Why? Refers to managing a singular product or product line, even if responsibilities span multiple products.
Risk Manager

It cannot be Product Managers it Has to be Product Manager
It cannot be Data Scientists it has to be Data Scientist

Why? Refers to the singular domain of "risk" management as a collective concept.
</Edge Case Examples>
We are covering these examples to make it 100 percent accurate
Make sure you understand the above edge case example and handle them accordingly

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""

prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt1_),
    ]
)

chains.append(prompt1 | llm_reply)


prompt2_ = """
So the question is  "And is this position full-time, part-time, or contract?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be a something like full time, part time or contract only and nothing illogical or irrelevant.
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. It might be short forms like FTE PTE etc.
5. You can allow any kind of typo mistakes or spelling mistakes dont be too strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict

On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question  and check if the answer is not logical or jubbarish not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back


Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""

prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt2_),
    ]
)

chains.append(prompt2 | llm_reply)


prompt3_ = """
So the question is  "Which department will this role be in?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be a something related to Department only and nothing illogical or irrelevant.
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question  and check if the answer is not logical or jubbarish not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.


You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""


prompt3 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt3_),
    ]
)

chains.append(prompt3 | llm_reply)


prompt4_ = """
So the question is  "Who will this person report to?""
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be a something related to a Reporting only.
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. It cannot be short forms because one short form can have multiple meanings to express those full forms as well
5.You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question  and check if the answer is not logical or jubbarish not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back


Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""

prompt4 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt4_),
    ]
)

chains.append(prompt4 | llm_reply)


prompt5_ = """ 
So the question is  "Where is this position based at?""
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be a city along with country or maybe only city. It cannot be a continent or only a country something which is completely illoggical.
Make sure it can be any city be it a small city like Vatican City as well. It should allow cities of any country be it anything.
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question and check if the answer is not logical or jubbarish not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
 """

prompt5 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt5_),
    ]
)

chains.append(prompt5 | llm_reply)


prompt6_ = """ 
So the question is  "Can you talk about the work arrangement for this role? (Hybrid, WFH, In Office, On Field Etc.)"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be a a proper answer like in office, on field or hybrid or work from home or shortcuts but nothing irrelevant.
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is and check if the answer is not logical or jubbarish not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back


Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
 """


prompt6 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt6_),
    ]
)


chains.append(prompt6 | llm_reply)

prompt7_ = """ 
So the question is  "Let's talk about the key responsibilities. Can you describe the main duties this person will handle?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be key responsibilities related to what is required from an employee only and nothing more than that.
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is and check if the answer is not logical or jubbarish not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back


Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
 """


prompt7 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt7_),
    ]
)

chains.append(prompt7 | llm_reply)


prompt8_ = """ 
So the question is  "Would you like to add any additional tasks or responsibilities?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be Yes or No or yes along with additional requirements.
In this case you need to accept informal Yes and No as well such as nopes, nah, yeah, yups etc.

Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. Verify if the answer is something like just yes without mentioning any requirements in that case we need to ask back some follow up questions so that the user can type back the requiements
5. If it has yes along with the extra info then score should be 1 only but if its only yes then 2
6. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right, 2 if a follow up question is supposed to be asked
Check if the answer does not have yes or no but it is giving some information related to the question asked then score should still come as 1
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.
Follow_up: If the answer is yes to this specific questions then a follow up questions should be asked something like "So what are those additional tasks or responsibilities" . On the basis of this refine something of your own and give me back a follow up question only in the case where someone is answering yes to this quesiton. In case answer is no or nah then score will be 0 and follow up can be Null
The score should be 1 if user answers the addtional tasks or responsibilities even without saying yes

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is  and check if the answer is not logical or jubbarish not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.


You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
 """


prompt8 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt8_),
    ]
)

chains.append(prompt8 | llm_follow_up)


prompt9_ = """ 
So the question is "What are the specific long-term goals and expectations from this position?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related goals and expectations only. verify very carefully if it isn't something illogical
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is  and check if the answer is not logical or jubbarish not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose

"""

prompt9 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt9_),
    ]
)

chains.append(prompt9 | llm_reply)

prompt10_ = """ 
So the question is "What is the immediate challenge that a new hire would face in this position?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to challenges only. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is  and check if the answer is not logical or jubbarish not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""


prompt10 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt10_),
    ]
)

chains.append(prompt10 | llm_reply)

prompt11_ = """ 
So the question is "What defines success in this role?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to success in the role. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which isand check if the answer is not logical or jubbarish or not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back


Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""

prompt11 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt11_),
    ]
)


chains.append(prompt11 | llm_reply)

prompt12_ = """ 
So the question is "How does this role align with and support the company's overarching strategic objectives?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to support the company's overarching strategic objectives?. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is and check if the answer is not logical or jubbarish or not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose

"""


prompt12 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt12_),
    ]
)


chains.append(prompt12 | llm_reply)

prompt13_ = """ 
So the question is "How frequently will this individual collaborate with teams from other departments, such as marketing, sales, or customer success?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to question only and nothing else. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is  and check if the answer is not logical or jubbarish or not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""

prompt13 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt13_),
    ]
)


chains.append(prompt13 | llm_reply)

prompt14_ = """ 
So the question is "Who are the key stakeholders this person will regularly interact with outside the immediate team? "

and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to the question and nothing apart from that. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question and check if the answer is not logical or jubbarish or not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose

"""

prompt14 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt14_),
    ]
)


chains.append(prompt14 | llm_reply)


prompt15_ = """ 
So the question is "Let's cover the skills and qualifications now. What's the minimum years of experience required for this role?"

and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to the question and nothing apart from that. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right. If the years of experience is below 60 then dont be too logical and just give score 1
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

IF someone is giving a range then he or she needs to put either 
1. hyphen for example (2-4) or 2. "to" for example (1 to 3) . Spaces aren't allowed and directly make the score 0

If someone is giving years of experience more than 60 then directly make the score 0 becuase that is not possible at all so it has to be less than 60 years

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question and check if the answer is not logical or jubbarish or not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose

Make sure if the years of experience is less than 60 then you dont need to give a score 0 you can give 1
"""

prompt15 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt15_),
    ]
)


chains.append(prompt15 | llm_reply)

prompt16_ = """ 
So the question is  "Great! Any specific educational background or certifications needed?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be Yes or No or yes along with educational background or certification needed.
In this case you need to accept informal Yes and No as well such as nopes, nah, yeah, yups etc.

Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. Verify if the answer is something like just yes without mentioning any requirements in that case we need to ask back some follow up questions so that the user can type back the requiements
5. If it has yes along with the extra info then score should be 1 only but if its only yes then 2
6. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict

On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right, 2 if a follow up question is supposed to be asked
Check if the answer does not have yes or no but it is giving some information related to the question asked then score should still come as 1
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.
Follow_up: If the answer is yes to this specific questions then a follow up questions should be asked something like "So what are the additional certifications you need?" . On the basis of this refine something of your own and give me back a follow up question only in the case where someone is answering yes to this quesiton. In case answer is no or nah then score will be 0 and follow up can be Null
The score should be 1 if user answers the educational background or certifications needed even without saying yes

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is  and check if the answer is not logical or jubbarish not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
 """


prompt16 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt16_),
    ]
)


chains.append(prompt16 | llm_follow_up)

prompt17_ = """ 
So the question is  "Are there any skills you expect them to develop"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be Yes or No or yes along with educational skills needed.
In this case you need to accept informal Yes and No as well such as nopes, nah, yeah, yups etc.

Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. Verify if the answer is something like just yes without mentioning any extra skills in that case we need to ask back some follow up questions so that the user can type back the requiements
5. If it has yes along with the extra info then score should be 1 only but if its only yes then 2
6. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict

On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right, 2 if a follow up question is supposed to be asked
Check if the answer does not have yes or no but it is giving some information related to the question asked then score should still come as 1
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.
Follow_up: If the answer is yes to this specific questions then a follow up questions should be asked something like "So what are the additional certifications you need?" . On the basis of this refine something of your own and give me back a follow up question only in the case where someone is answering yes to this quesiton. In case answer is no or nah then score will be 0 and follow up can be Null
The score should be 1 if user answers the extra skills needed even without saying yes

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is  and check if the answer is not logical or jubbarish not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
 """

prompt17 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt17_),
    ]
)

chains.append(prompt17 | llm_follow_up)

prompt18_ = """ 
So the question is  "And are there any must-have technical skills or soft skills?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be Yes or No or yes along with technical or soft skills needed.
In this case you need to accept informal Yes and No as well such as nopes, nah, yeah, yups etc.

Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. Verify if the answer is something like just yes without mentioning any extra skills in that case we need to ask back some follow up questions so that the user can type back the requiements
5. If it has yes along with the extra info then score should be 1 only but if its only yes then 2
6. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right, 2 if a follow up question is supposed to be asked
Check if the answer does not have yes or no but it is giving some information related to the question asked then score should still come as 1
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.
Follow_up: If the answer is yes to this specific questions then a follow up questions should be asked something like "So what are the additional certifications you need?" . On the basis of this refine something of your own and give me back a follow up question only in the case where someone is answering yes to this quesiton. In case answer is no or nah then score will be 0 and follow up can be Null
The score should be 1 if user answers the technical skills or soft skills needed even without saying yes
Please allow to move forward for simple spelling mistakes and dont be very strict
Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is  and check if the answer is not logical or jubbarish not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back


Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
 """

prompt18 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt18_),
    ]
)

chains.append(prompt18 | llm_follow_up)

prompt19_ = """ 
So the question is "What type of working style thrives for this role?"

and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to the question and nothing apart from that. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question and check if the answer is not logical or jubbarish or not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""

prompt19 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt19_),
    ]
)

chains.append(prompt19 | llm_reply)

prompt20_ = """ 
So the question is  "Are there any additional qualifications or skills that would be a bonus?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be Yes or No or yes along with technical or soft skills needed.
In this case you need to accept informal Yes and No as well such as nopes, nah, yeah, yups etc.

Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. Verify if the answer is something like just yes without mentioning any extra skills or qualification in that case we need to ask back some follow up questions so that the user can type back the requiements
5. If it has yes along with the extra info then score should be 1 only but if its only yes then 2
6. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right, 2 if a follow up question is supposed to be asked
Check if the answer does not have yes or no but it is giving some information related to the question asked then score should still come as 1
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.
Follow_up: If the answer is yes to this specific questions then a follow up questions should be asked something like "So what are the additional certifications you need?" . On the basis of this refine something of your own and give me back a follow up question only in the case where someone is answering yes to this quesiton. In case answer is no or nah then score will be 0 and follow up can be Null
The score should be 1 if user answers the  additional qualifications or skills needed even without saying yes
Please allow to move forward for simple spelling mistakes and dont be very strict

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is  and check if the answer is not logical or jubbarish not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
 """

prompt20 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt20_),
    ]
)

chains.append(prompt20 | llm_follow_up)


prompt21_ = """ 
So the question is  "Is there any preferred candidate background that would fit the role best?"
and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be Yes or No or yes along with technical or soft skills needed.
In this case you need to accept informal Yes and No as well such as nopes, nah, yeah, yups etc.

Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. Verify if the answer is something like just yes without mentioning any extra candidate background in that case we need to ask back some follow up questions so that the user can type back the requiements
5. If it has yes along with the extra info then score should be 1 only but if its only yes then 2
6. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict

On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right, 2 if a follow up question is supposed to be asked
Check if the answer does not have yes or no but it is giving some information related to the question asked then score should still come as 1
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.
Follow_up: If the answer is yes to this specific questions then a follow up questions should be asked something like "So what are the additional certifications you need?" . On the basis of this refine something of your own and give me back a follow up question only in the case where someone is answering yes to this quesiton. In case answer is no or nah then score will be 0 and follow up can be Null
The score should be 1 if user answers the candidate background needed even without saying yes

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question which is  and check if the answer is not logical or jubbarish not sensible logically. Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back
Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
 """


prompt21 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt21_),
    ]
)


chains.append(prompt21 | llm_follow_up)

prompt22_ = """ 
So the question is "What opportunities for learning and growth does this role offer?"

and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to the question and nothing apart from that. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict

On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question and check if the answer is not logical or jubbarish or not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back


Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""

prompt22 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt22_),
    ]
)

chains.append(prompt22 | llm_reply)

prompt23_ = """ 
So the question is "What benefits will you offer with this role?"

and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to the question and nothing apart from that. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4.You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict

On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question and check if the answer is not logical or jubbarish or not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""

prompt23 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt23_),
    ]
)

chains.append(prompt23 | llm_reply)


prompt24_ = """ 
So the question is "What kind of guidance or mentorship can the new hire expect?"

and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to the question and nothing apart from that. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict

On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question and check if the answer is not logical or jubbarish or not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""

prompt24 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt24_),
    ]
)

chains.append(prompt24 | llm_reply)

prompt25_ = """ 
So the question is "How would you describe your management style?"

and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to the question and nothing apart from that. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict

On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Check if the answer does not have yes or no but it is giving some information related to the question asked then score should still come as 1
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.
Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 

Important Points while replying:
- Carefully understand what went wrong on the answer based on the question and check if the answer is not logical or jubbarish or not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose
"""

prompt25 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt25_),
    ]
)

chains.append(prompt25 | llm_reply)

prompt26_ = """ 
So the question is "And what's the work culture like on the team?"

and the answer given by the user is "{answer}"

Now you need to analyse the answer very intelligently and understand if the answer is right or not. The answer should be something related to the question and nothing apart from that. Now you cant put something which illogical or not sensible so make sure of that as well
Additional Key Points 
1. It cannot be some jubbarish data.
2. Verify if the answer is coherent and not gibberish or nonsense.
3. Ensure the answer is relevant to the question and not completely invalid or unrelated.
4. You can allow any kind of typo mistakes or spelling mistakes dont be strict on that at all and clear to the next question with score 1
Please allow to move forward for simple spelling mistakes and dont be very strict

On the basis of this you need to return me back 2 things
Score: 0 If the answer is not related to the question or not logical enough, 1 if the answer is relevant and sticks to the question and is also logically right
Reply: This should be an empty string if score is 1 since its relevance, Now if the score is 0 then you need to reply very carefully and very logically.

Key point: Give the reply based on the answer the human has given and it should relate to that. Dont keep a fixed answer if user is not understanding or anything help them 
Important Points while replying:
- Carefully understand what went wrong on the answer based on the question and check if the answer is not logical or jubbarish or not sensible logically . Even if it is relevant and it doesn't seem logically right still mark the score 0
- Reflect in accordance to that and give a reply memntioning what went wrong and what he can answer to get it right basically give 2 or 3 examples
- Dont make the response too verbose and keep it concise so the user does not need to read the message much.
If user asks back something for example "can you suggest something?" then dont mention that the answer is wrong or something reply sweetly based on the answer
On the basis of these important points give a reply that too only if the score is 0.

You give reply based on answer and question very acurately. 
For example if someone says thank you then you say politely welcome and and ask the question back
if someone is not understanding the question then you will help them understand it well and give a reply nicely as a human and then aask the question back politely

So you need to have very good way of replying just like a human based on the question which you asked and the answer the human had provided
Make sure you address the problem which the human is having in the answer sweetly like an agentic chatbot and then ask the question back

Very important Points:
Also make sure when user is asking for suggestions then you dont write  "It seems like you are asking for suggestion or examples or whatever they are asking." it should not also say "It seems like your answer is irrelevant to the question" Note: This happens only if user is asking suggestions or examples
It should directly and clearly reply here are some suggestions or examples in the reply. 
If it says thank you then say welcome and then ask the question back
Make sure the reply is concise and not verbose

"""

prompt26 = ChatPromptTemplate.from_messages(
    [
        ("system", eval_system_prompt),
        MessagesPlaceholder("chat_history"),  # This allows passing in previous messages
        ("user", prompt26_),
    ]
)


chains.append(prompt26 | llm_reply)


question_chain_pair = {}
for i in range(len(questions_list)):
    question_chain_pair[questions_list[i]] = chains[i]


question_chain_pair[
    "What's the minimum years of experience required for this role?"
] = prompt15 | llm_reply
