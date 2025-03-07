from fastapi import FastAPI, Response, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from typing import Optional, Dict, List
from datetime import datetime
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_together import Together
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
from sse_starlette.sse import EventSourceResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Coy Chatbot", description="A chatbot with personality switching capabilities")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="Templates")

class ChatMessage(BaseModel):
    message: str
    role: str = "user"

class ChatbotState:
    def __init__(self):
        self.conversations: Dict[str, Dict] = {}
        self.bot_conversation_active = False
        self.initialize_models()

    def initialize_models(self):
        """Initialize the language models and conversation chains"""
        try:
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY not found in environment variables")

            self.llm1 = Together(
                model=os.getenv("MODEL1_NAME"),
                temperature=float(os.getenv("TEMPERATURE", 0.0)),
                together_api_key=api_key
            )

            self.llm2 = Together(
                model=os.getenv("MODEL2_NAME"),
                temperature=float(os.getenv("TEMPERATURE", 0.0)),
                together_api_key=api_key
            )

            self.memory1 = ConversationBufferMemory()
            self.memory2 = ConversationBufferMemory()
            
            self.conversation_chain1 = ConversationChain(llm=self.llm1, memory=self.memory1)
            self.conversation_chain2 = ConversationChain(llm=self.llm2, memory=self.memory2)
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize chatbot models")

    async def generate_bot_conversation(self, topic: str, rounds: int = 3):
        """Generate a conversation between the two bots"""
        try:
            self.bot_conversation_active = True
            conversation_history = []
            current_topic = topic

            for _ in range(rounds):
                # Bot 1's turn (Happy personality)
                response1 = self.conversation_chain1.predict(
                    input=f"As a happy and enthusiastic personality, continue the conversation about {current_topic}. Keep your response concise and engaging."
                )
                conversation_history.append({"content": response1, "role": "bot1", "mood": "happy"})
                yield conversation_history[-1]
                await asyncio.sleep(2)  # Add natural pause between responses

                # Bot 2's turn (Sassy personality)
                response2 = self.conversation_chain2.predict(
                    input=f"As a sassy and playful personality, respond to: '{response1}' while discussing {current_topic}. Keep your response concise and witty."
                )
                conversation_history.append({"content": response2, "role": "bot2", "mood": "sassy"})
                yield conversation_history[-1]
                await asyncio.sleep(2)

                # Update the topic based on the conversation
                current_topic = f"the previous exchange: '{response2}'"

        except Exception as e:
            logger.error(f"Error in bot conversation: {str(e)}")
            yield {"error": str(e)}
        finally:
            self.bot_conversation_active = False

chatbot_state = ChatbotState()

@app.get("/")
async def read_root():
    return {"status": "ok", "message": "Coy Chatbot API is running"}

@app.post("/chatbot")
@limiter.limit("5/minute")
async def chatbot(message: ChatMessage):
    try:
        # Validate input
        if not message.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Process message with first chatbot (happy personality)
        response1 = chatbot_state.conversation_chain1.predict(
            input=f"Respond to this message in a happy and enthusiastic way: {message.message}"
        )

        # Process first response with second chatbot (sassy personality)
        response2 = chatbot_state.conversation_chain2.predict(
            input=f"Respond to this message in a sassy and playful way, referencing what was just said: {response1}"
        )

        # Log the interaction
        logger.info(f"Processed message: {message.message[:50]}...")

        return JSONResponse({
            "responses": [
                {"content": response1, "role": "bot1", "mood": "happy"},
                {"content": response2, "role": "bot2", "mood": "sassy"}
            ],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bot-conversation")
async def bot_conversation(topic: str = "a random interesting topic"):
    """Endpoint for bot-to-bot conversation using Server-Sent Events"""
    if chatbot_state.bot_conversation_active:
        raise HTTPException(status_code=409, detail="A bot conversation is already in progress")

    async def event_generator():
        async for response in chatbot_state.generate_bot_conversation(topic):
            if "error" in response:
                yield {
                    "event": "error",
                    "data": response["error"]
                }
                return
            yield {
                "event": "message",
                "data": {
                    **response,
                    "timestamp": datetime.now().isoformat()
                }
            }

    return EventSourceResponse(event_generator())

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)