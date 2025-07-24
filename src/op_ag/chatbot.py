# AIzaSyDOcsWpGNFL77WmZECvc0-tl3F04lTIaGY
import chainlit as cl
import os 
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv
from agents.tool import function_tool

load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

# step1
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# step2
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client= provider
)

# config defined at run level
run_config = RunConfig(
    model = model,
    model_provider = provider,
    tracing_disabled= True
)

@function_tool("get_weather")
def get_weather(location: str) -> str:
    return f"The weather in {location} is 22C"

# step3 Agent
agent1 = Agent(
    instructions = "Your are helpful assistant that can answer questions, use get_weather tool to fetch weather of any location, you reply very consicely",
    name = "Hospital Support Agent",
    tools=[get_weather]
)

# step4 Run
result = Runner.run_sync(
    agent1,
    input = "What is the temperature of India?",
    run_config=run_config
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Hello, I am Expedey support agent. How can I help you?").send()



@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role":"user", "content": message.content})

    result = await Runner.run(
    agent1,
    input = history,
    run_config=run_config
)

    history.append({"role":"assistant","content":result.final_output})
    await cl.Message(
        content=result.final_output,
    ).send()

