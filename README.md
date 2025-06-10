# 🦷 Dentaly RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and OpenAI that answers dental care questions based on content from the Dentaly US website.

## Features

- **Semantic Search**: Uses OpenAI embeddings to find relevant content
- **Streaming Responses**: Real-time streaming responses from GPT-4
- **Source Citations**: Provides source URLs for transparency
- **User-Friendly Interface**: Clean Streamlit interface with chat history
- **Flexible API Key Input**: Supports both environment secrets and user input

## Demo

🚀 **[Try the live demo here](https://your-app-name.railway.app)** (Replace with your Railway URL)

## How It Works

1. **Knowledge Base**: Pre-processed embeddings from Dentaly US website content
2. **Query Processing**: User questions are converted to embeddings
3. **Retrieval**: Most relevant documents are found using cosine similarity
4. **Generation**: GPT-4 generates responses based on retrieved context
5. **Citation**: Source URLs are provided with each response

## Quick Start

### Option 1: Use the Live Demo
Visit the deployed version and enter your OpenAI API key in the sidebar.

### Option 2: Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Pablo751/dental-chatbot.git
   cd dental-chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Enter your OpenAI API Key** in the sidebar when prompted.

## Configuration

### For Local Development
You can either:
- Enter your API key through the Streamlit interface (recommended for testing)
- Create a `.streamlit/secrets.toml` file:
  ```toml
  OPENAI_API_KEY = "your-api-key-here"
  ```

### For Production Deployment
Set the `OPENAI_API_KEY` environment variable in your deployment platform.

## Files Structure

```
├── app.py                          # Main Streamlit application
├── create_embeddings.py            # Script to generate embeddings
├── dentaly_us_embeddings.pkl       # Pre-computed embeddings (2.7MB)
├── Dentaly URLS - Dentaly US.csv   # Source URLs and content
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Deployment

### Railway Deployment

1. **Fork this repository** to your GitHub account

2. **Connect to Railway**:
   - Go to [Railway](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your forked repository

3. **Set Environment Variables** (Optional):
   - Add `OPENAI_API_KEY` if you want to provide a default key
   
4. **Deploy**: Railway will automatically detect the Streamlit app and deploy it

### Other Platforms
- **Streamlit Cloud**: Connect your GitHub repo directly
- **Heroku**: Add `setup.sh` and `Procfile` (not included)
- **Vercel**: May require additional configuration

## How to Get an OpenAI API Key

1. Visit [OpenAI's website](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste it into the Streamlit sidebar

## Technical Details

- **Embedding Model**: `text-embedding-3-small`
- **Chat Model**: `gpt-4o`
- **Vector Database**: In-memory NumPy arrays with cosine similarity
- **Frontend**: Streamlit with streaming chat interface

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For questions or suggestions, please open an issue on GitHub.

---

Made with ❤️ for better dental health education 