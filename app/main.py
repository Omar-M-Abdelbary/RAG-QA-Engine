from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

from app.routes.rag_routes import router as rag_router
from app.routes.admin_routes import router as admin_router
from app.routes.evaluation_routes import router as evaluation_router
from app.infrastructure.database.session import init_db
from app.utils.logger import get_logger



logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown"""
    # Startup
    logger.info("=" * 60)
    logger.info(" RAG API Starting...")
    logger.info("=" * 60)
    
    # Initialize database
    try:
        init_db()
        logger.info(" Database initialized")
    except Exception as e:
        logger.error(f" Database initialization failed: {e}")
    
    logger.info(" API Documentation: http://localhost:8000/docs")
    logger.info(" Health Check: http://localhost:8000/api/v1/health")
    logger.info(" Statistics: http://localhost:8000/api/v1/stats")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info(" RAG API Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="RAG Question Answering System",
    description="""
    A production-ready RAG (Retrieval-Augmented Generation) system for answering questions 
    using semantic search and LLM generation.
    
    ## Features
    
    * **Semantic Search**: Uses FAISS for efficient similarity search
    * **LLM Integration**: Powered by Groq for answer generation
    * **Caching**: Smart caching to reduce latency and costs
    * **Analytics**: Track queries and performance metrics
    * **Validation**: Quality scoring for generated answers
    
    ## Endpoints
    
    * `/ask-question` - Ask questions and get AI-generated answers
    * `/health` - Check system health
    * `/stats` - View system statistics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    rag_router, 
    prefix="/api/v1"
)
app.include_router(
    admin_router,
    prefix="/api/v1/admin"
)

app.include_router(
    evaluation_router,
    prefix="/api/v1/evaluation"
)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Welcome to RAG Question Answering API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "stats": "/api/v1/stats"
    }

# Custom exception handler (optional - for better error messages)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom validation error handler"""
    errors = exc.errors()
    
    # Format errors in a user-friendly way
    formatted_errors = []
    for error in errors:
        field = " -> ".join(str(x) for x in error['loc'][1:])  # Skip 'body'
        formatted_errors.append({
            "field": field,
            "message": error['msg']
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Invalid input provided",
            "details": formatted_errors
        }
    )