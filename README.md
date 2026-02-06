# LangGraph Multi-Utility Chatbot

A comprehensive chatbot application built with LangGraph, Streamlit, and multiple AI backends, featuring conversation persistence, tool integration, and RAG (Retrieval-Augmented Generation) capabilities.

## Features

### Core Functionality
- **Multiple AI Backends**: Support for both Hugging Face (Llama 3) and OpenAI models
- **Conversation Persistence**: SQLite-based conversation storage with thread management
- **Multiple Frontend Variants**: Various Streamlit interfaces for different use cases
- **Tool Integration**: Built-in tools for web search, calculations, and stock prices
- **RAG Support**: PDF document ingestion and question-answering capabilities

### Available Tools
- **Web Search**: DuckDuckGo integration for real-time information
- **Calculator**: Basic arithmetic operations (add, subtract, multiply, divide)
- **Stock Prices**: Real-time stock data via Alpha Vantage API
- **PDF RAG**: Upload and query PDF documents using vector embeddings

## Project Structure

```
├── backend.py              # Basic chatbot with in-memory storage
├── backend_database.py     # Chatbot with SQLite persistence
├── backend_tools.py        # Chatbot with integrated tools
├── backend_rag.py          # Full-featured backend with RAG and all tools
├── streamlit_frontend.py   # Basic Streamlit interface
├── streamlit_database.py   # Interface with conversation history
├── streamlit_rag.py        # Full-featured interface with PDF upload
├── streamlit_streaming.py  # Streaming response interface
├── streamlit_threading.py  # Multi-thread conversation management
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
└── chatbot.db            # SQLite database for conversations
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file with the following variables:
   ```env
   # OpenAI Configuration (for RAG features)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Hugging Face Configuration
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   
   # LangChain Tracing (optional)
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   LANGCHAIN_PROJECT=Chatbot Project
   ```

## Usage

### Running Different Variants

Choose the appropriate backend and frontend combination based on your needs:

#### Basic Chatbot
```bash
streamlit run streamlit_frontend.py
```
- Uses `backend.py`
- In-memory conversations
- Simple chat interface

#### Persistent Conversations
```bash
streamlit run streamlit_database.py
```
- Uses `backend_tools.py`
- SQLite conversation storage
- Multiple conversation threads
- Integrated tools (search, calculator, stock prices)

#### Full-Featured RAG Chatbot
```bash
streamlit run streamlit_rag.py
```
- Uses `backend_rag.py`
- All features included
- PDF document upload and querying
- Vector embeddings with FAISS
- All integrated tools

#### Streaming Interface
```bash
streamlit run streamlit_streaming.py
```
- Real-time streaming responses
- Uses basic backend

## API Keys Required

### Essential
- **Hugging Face Token**: Required for Llama 3 model access
- **OpenAI API Key**: Required for RAG features and embeddings

### Optional
- **Alpha Vantage API Key**: For stock price functionality (free tier available)
- **LangChain API Key**: For conversation tracing and monitoring

## Key Components

### Backends

1. **backend.py**: Minimal implementation with in-memory storage
2. **backend_database.py**: Adds SQLite persistence
3. **backend_tools.py**: Includes web search, calculator, and stock tools
4. **backend_rag.py**: Complete implementation with PDF RAG capabilities

### Frontend Interfaces

1. **streamlit_frontend.py**: Basic chat interface
2. **streamlit_database.py**: Multi-thread conversation management
3. **streamlit_rag.py**: Full-featured interface with PDF upload
4. **streamlit_streaming.py**: Real-time streaming responses
5. **streamlit_threading.py**: Advanced thread management

### Tools Available

- **DuckDuckGo Search**: Web search functionality
- **Calculator**: Basic arithmetic operations
- **Stock Price Lookup**: Real-time stock data
- **RAG Tool**: Query uploaded PDF documents

## Configuration

### Model Configuration
- **Hugging Face Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **OpenAI Embeddings**: `text-embedding-3-large`
- **Vector Store**: FAISS for document embeddings
- **Database**: SQLite for conversation persistence

### Customization
- Modify model parameters in backend files
- Adjust chunk size and overlap in RAG configuration
- Configure tool parameters and API endpoints
- Customize Streamlit interface layouts

## Troubleshooting

### Common Issues
1. **Missing API Keys**: Ensure all required environment variables are set
2. **Model Loading**: Verify Hugging Face token has proper permissions
3. **Database Errors**: Check SQLite file permissions and disk space
4. **PDF Processing**: Ensure uploaded files are valid PDF format

### Performance Tips
- Use smaller models for faster responses
- Adjust chunk sizes for better RAG performance
- Enable conversation caching for repeated queries
- Monitor API usage to avoid rate limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different backend/frontend combinations
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review environment variable configuration
- Ensure all dependencies are properly installed
- Verify API key permissions and quotas