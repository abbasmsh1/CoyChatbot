# Coy Chatbot

A unique chatbot application featuring two AI personalities that interact with users and each other. One personality is happy and enthusiastic, while the other is sassy and playful, creating engaging and entertaining conversations.

## Features

- Dual AI personalities with distinct conversation styles
- Real-time chat interface with modern UI
- Message history with timestamps
- Rate limiting to prevent abuse
- Error handling and loading states
- Responsive design for all devices

## Prerequisites

- Python 3.8 or higher
- Together AI API key

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CoyChatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Together AI API key:
```
TOGETHER_API_KEY=your_api_key_here
HOST=127.0.0.1
PORT=8000
MODEL1_NAME=mistralai/Mixtral-8x7B-Instruct-v0.1
MODEL2_NAME=META-LLAMA/LLAMA-2-70B-CHAT-HF
TEMPERATURE=0.0
```

## Running the Application

1. Start the server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

## Usage

1. Type your message in the input field at the bottom of the chat interface
2. Press Enter or click the Send button to send your message
3. Wait for both AI personalities to respond
4. Continue the conversation as desired

## Rate Limiting

The application includes rate limiting to prevent abuse:
- 5 messages per minute per IP address

## Error Handling

The application includes comprehensive error handling:
- API errors are displayed to the user
- Network issues are caught and displayed
- Invalid responses are handled gracefully

## Security

- API keys are stored in environment variables
- CORS protection is enabled
- Rate limiting prevents abuse
- Input validation is implemented

## License

MIT License - feel free to use this project for any purpose.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 