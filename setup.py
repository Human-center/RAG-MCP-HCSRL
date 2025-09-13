#!/usr/bin/env python3
"""
RAG System Setup Script

This script helps set up the RAG system environment and verify that all components
are working correctly.

Usage:
    python setup.py
    python setup.py --check-only
    python setup.py --install-deps
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )


def check_python_version():
    """Check Python version compatibility."""
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3, 8):
        logger.error(f"Python 3.8+ required, but you have {sys.version_info.major}.{sys.version_info.minor}")
        return False
    
    logger.info(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ✓")
    return True


def check_virtual_environment():
    """Check if running in virtual environment."""
    logger = logging.getLogger(__name__)
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.info("Virtual environment detected ✓")
        return True
    else:
        logger.warning("No virtual environment detected - consider using one")
        return False


def install_dependencies():
    """Install dependencies from requirements.txt."""
    logger = logging.getLogger(__name__)
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        return False
    
    try:
        logger.info("Installing dependencies from requirements.txt...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True, text=True)
        
        logger.info("Dependencies installed successfully ✓")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        ("torch", "PyTorch"),
        ("sentence_transformers", "Sentence Transformers"),
        ("chromadb", "ChromaDB"),
        ("PyPDF2", "PyPDF2"),
        ("fitz", "PyMuPDF"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
        ("nltk", "NLTK")
    ]
    
    missing_packages = []
    
    for package, display_name in required_packages:
        try:
            __import__(package)
            logger.info(f"{display_name}: ✓")
        except ImportError:
            logger.error(f"{display_name}: ✗ (missing)")
            missing_packages.append(display_name)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        return False
    
    logger.info("All required dependencies found ✓")
    return True


def check_huggingface_auth():
    """Check Hugging Face authentication."""
    logger = logging.getLogger(__name__)
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Try to get user info
        user_info = api.whoami()
        if user_info:
            logger.info(f"Hugging Face authentication: ✓ (logged in as {user_info['name']})")
            return True
        else:
            logger.warning("Hugging Face authentication: ⚠ (not logged in)")
            return False
            
    except Exception as e:
        logger.warning(f"Hugging Face authentication check failed: {e}")
        logger.info("You may need to run 'huggingface-cli login' to access EmbeddingGemma")
        return False


def check_gpu_availability():
    """Check GPU availability for PyTorch."""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"CUDA GPU: ✓ ({gpu_count} device(s), {gpu_name})")
            return True
        else:
            logger.info("CUDA GPU: ⚠ (not available, will use CPU)")
            return False
            
    except ImportError:
        logger.error("PyTorch not available for GPU check")
        return False


def test_basic_functionality():
    """Test basic functionality of the system."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test ChromaDB initialization
        logger.info("Testing ChromaDB...")
        import chromadb
        from chromadb.config import Settings
        
        # Create a temporary client
        temp_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create a test collection
        test_collection = temp_client.create_collection("test_collection")
        test_collection.add(
            documents=["Test document"],
            ids=["test_1"],
            metadatas=[{"test": True}]
        )
        
        # Test query
        results = test_collection.query(query_texts=["Test query"], n_results=1)
        if results and results['documents']:
            logger.info("ChromaDB: ✓")
        else:
            logger.error("ChromaDB: ✗ (query failed)")
            return False
        
        # Test EmbeddingGemma (if available)
        logger.info("Testing EmbeddingGemma...")
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer("google/embeddinggemma-300M").to(device=device)
            
            # Test embedding generation
            test_text = "This is a test sentence."
            embedding = model.encode([test_text])
            
            if embedding is not None and len(embedding) > 0:
                logger.info("EmbeddingGemma: ✓")
            else:
                logger.error("EmbeddingGemma: ✗ (embedding generation failed)")
                return False
                
        except Exception as e:
            logger.warning(f"EmbeddingGemma: ⚠ (test failed: {e})")
            logger.info("You may need to authenticate with Hugging Face: huggingface-cli login")
            return False
        
        # Test PDF processing
        logger.info("Testing PDF processing...")
        try:
            import PyPDF2
            import fitz  # PyMuPDF
            logger.info("PDF processing libraries: ✓")
        except ImportError as e:
            logger.error(f"PDF processing libraries: ✗ ({e})")
            return False
        
        logger.info("Basic functionality test: ✓")
        return True
        
    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        return False


def create_directory_structure():
    """Create necessary directory structure."""
    logger = logging.getLogger(__name__)
    
    directories = [
        "./chroma_db",
        "./documents"  # For sample PDFs
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory exists: {directory}")
    
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="RAG System Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full setup and verification
    python setup.py
    
    # Only check current status
    python setup.py --check-only
    
    # Install dependencies
    python setup.py --install-deps
    
    # Skip functionality tests
    python setup.py --skip-tests
        """
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check current status, don't install anything"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install dependencies from requirements.txt"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip functionality tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    print("=== RAG System Setup ===\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    check_virtual_environment()
    
    # Install dependencies if requested
    if args.install_deps or not args.check_only:
        if not check_dependencies():
            if not args.check_only:
                logger.info("Installing missing dependencies...")
                if not install_dependencies():
                    sys.exit(1)
            else:
                logger.error("Missing dependencies - run with --install-deps to install")
                sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        if not args.install_deps and not args.check_only:
            logger.error("Dependencies missing - consider running with --install-deps")
        sys.exit(1 if args.check_only else 0)
    
    # Check Hugging Face authentication
    check_huggingface_auth()
    
    # Check GPU availability
    check_gpu_availability()
    
    # Create directory structure
    if not args.check_only:
        create_directory_structure()
    
    # Test basic functionality
    if not args.skip_tests and not args.check_only:
        if not test_basic_functionality():
            logger.error("Basic functionality tests failed")
            sys.exit(1)
    
    print("\n=== Setup Summary ===")
    print("✓ Python version compatible")
    print("✓ Dependencies installed")
    print("✓ Directory structure created")
    
    if not args.skip_tests and not args.check_only:
        print("✓ Basic functionality verified")
    
    print("\n=== Next Steps ===")
    print("1. Place PDF files in the ./documents directory")
    print("2. Initialize ChromaDB: python init_chroma.py")
    print("3. Ingest PDFs: python ingest_pdfs.py --input-dir ./documents")
    print("4. Test queries: python test_rag_query.py --query 'your query here'")
    
    if not check_huggingface_auth():
        print("\n⚠ Note: For EmbeddingGemma access, run: huggingface-cli login")


if __name__ == "__main__":
    main()