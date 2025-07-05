# Mira - AI Tutor

[![Built at Bolt](https://img.shields.io/badge/Built%20at-Bolt-blue?style=for-the-badge&logo=lightning)](https://bolt.new)

An intelligent AI tutor application that helps students understand documents through interactive chat and video conversations.

## Features

- **Document Upload & Processing**: Upload PDF documents for AI analysis
- **Interactive Chat**: Text-based conversations about document content
- **Video Chat**: Face-to-face conversations with AI tutor Mira
- **User Authentication**: Secure login with Clerk
- **Document History**: Access previously uploaded documents
- **Real-time Processing**: Fast document vectorization and AI responses

## Tech Stack

### Frontend
- **Next.js 13** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Modern UI components
- **Google OAuth** - Authentication and user management

### Backend
- **FastAPI** - High-performance Python web framework
- **SQLAlchemy** - Database ORM
- **Google Generative AI** - AI language model
- **LangChain** - AI application framework
- **FAISS** - Vector similarity search
- **Tavus API** - Video avatar generation

## Getting Started

### Prerequisites
- Node.js 18+ 
- Python 3.11+ (for local non-Docker development)
- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-tutor-app
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   ```

3. **Set up Python backend using Docker**
   
   The backend is containerized using Docker, which simplifies setup. Make sure you have Docker running on your machine.

4. **Environment Variables**
   
   Create `.env` files in both the root directory and the `api` directory with the required variables:
   
   **Root `/.env`:**
   ```env
   NEXT_PUBLIC_GOOGLE_CLIENT_ID=your_google_client_id
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```
   
   **API `/api/.env`:**
   ```env
   GOOGLE_CLIENT_ID=your_google_client_id
   GOOGLE_CLIENT_SECRET=your_google_client_secret
   JWT_SECRET_KEY=your_jwt_secret_key_change_in_production
   GOOGLE_API_KEY=your_google_api_key
   TAVUS_API_KEY=your_tavus_api_key
   TAVUS_REPLICA_ID=your_tavus_replica_id
   ```

   **To set up Google OAuth:**
   1. Go to [Google Cloud Console](https://console.cloud.google.com/)
   2. Create a new project or select existing one
   3. Enable Google+ API
   4. Create OAuth 2.0 credentials
   5. Add authorized origins: `http://localhost:3000` (and your production domain)
   6. Copy the client ID and secret to your `.env` files

5. **Run the application**
   
   **Start the backend (with Docker Compose):**
   ```bash
   cd api
   docker-compose up --build
   ```
   
   **Start the frontend (in a separate terminal):**
   ```bash
   npm run dev
   ```

6. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Usage

1. **Sign Up/Login**: Create an account or sign in using Google OAuth authentication
2. **Upload Document**: Upload a PDF file for AI processing
3. **Chat**: Start a text conversation about your document
4. **Video Chat**: Initiate a video conversation with AI tutor Mira
5. **History**: Access previously uploaded documents and conversations

## API Endpoints

- `POST /api/upload` - Upload and process documents
- `POST /api/chat` - Send chat messages
- `GET /api/documents` - Get user's documents
- `GET /api/chat-history/{document_name}` - Get chat history
- `POST /api/create_conversation` - Create video conversation
- `GET /api/video-call-usage` - Get video call usage stats

## Deployment

The application supports deployment on various platforms:

- **Frontend**: Vercel, Netlify, or any static hosting
- **Backend**: Railway, Heroku, or any Python hosting service
- **Database**: PostgreSQL, MySQL, or SQLite

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with ❤️ using [Bolt](https://bolt.new)
- Powered by Google Generative AI
- Video avatars by Tavus
- UI components by shadcn/ui