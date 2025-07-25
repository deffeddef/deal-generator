#!/usr/bin/env python3
"""
Main entry point for AIDG v2.5 Personalization Service
"""

import asyncio
import uvicorn
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.services.personalization_service import app
    from src.config.settings import config
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logging.basicConfig(
    level=getattr(logging, config.monitoring.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    
    logger.info("Starting AIDG v2.5 Personalization Service")
    logger.info(f"Service version: 2.5.0")
    logger.info(f"Configuration: {config.service}")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Check if running in development or production
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "development":
        logger.info("Running in development mode")
        uvicorn.run(
            "main:app",
            host=config.service.host,
            port=config.service.port,
            reload=True,
            log_level="info"
        )
    else:
        logger.info("Running in production mode")
        uvicorn.run(
            "main:app",
            host=config.service.host,
            port=config.service.port,
            workers=config.service.workers,
            log_level="info"
        )


if __name__ == "__main__":
    main()