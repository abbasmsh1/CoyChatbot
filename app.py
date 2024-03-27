from fastapi import FastAPI, Response
import os
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_together import Together

app = FastAPI()

def together_Ai(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0.0):
    """
    Create and initialize a Together AI language model.

    Parameters:
    - model_name (str, optional): The name of the Together AI language model.
    - temperature (float, optional): The parameter for randomness in text generation.
    - tokens (int, optional): The maximum number of tokens to generate.

    Returns:
    - llm (Together): The initialized Together AI language model.
    """

    api_key = "daacc8dc45f272f48e8571c2ff9bbccc7169541e632faa75e7efa12900cf2813"

    llm = Together(
        model=model_name,
        temperature=temperature,
        together_api_key=api_key
    )

    return llm

llm1 = together_Ai()
llm2 = together_Ai(model_name='META-LLAMA/LLAMA-2-70B-CHAT-HF')

memory1 = ConversationBufferMemory()
conversation_chain1 = ConversationChain(llm=llm1, memory=memory1)

memory2 = ConversationBufferMemory()
conversation_chain2 = ConversationChain(llm=llm2, memory=memory2)

mood1 = "happy"
mood2 = "angry"

@app.get("/chatbot")
def chatbot(input_message):
    global output1, output2, mood1, mood2

    # Pass output from one chatbot to the other with mood
    updated_message_prompt = f"Chatbot 1 (Mood: {mood1}) said: {output1}\nWhat should Chatbot 2 (Mood: {mood2}) say?"
    output2 = conversation_chain2.predict(input=output1)

    # Generate response from other chatbot
    updated_message_prompt = f"Chatbot 2 (Mood: {mood2}) said: {output2}\nWhat should Chatbot 1 (Mood: {mood1}) say?"
    output1 = conversation_chain1.predict(input=output2)

    # Swap moods for next iteration
    mood1, mood2 = mood2, mood1

    return {"output": output2}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)