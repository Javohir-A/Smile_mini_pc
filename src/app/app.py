import sys
import signal
import logging 
import asyncio
from dotenv import load_dotenv

from .camera_streaming_app import StreamApplication
from src.config.settings import AppConfig
from src.config import AppConfig
from src.di.dependencies import initialize_dependencies, DependencyContainer

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

async def main():
    """Main application entry point"""
    try:
        # Parse command line arguments
        env_version = "production"
        if len(sys.argv) > 1:
            versions = ["development", "production", "minipc-01", "minipc-02"]
            if sys.argv[1] not in versions:
                print(f"❌ Environment version must be one of: {versions}")
                sys.exit(1)
            env_version = sys.argv[1]
        
        # Load configuration from environment
        logger.info(f"🔧 Loading configuration for environment: {env_version}")
        load_dotenv(f'.env.{env_version}')
        config = AppConfig.from_env()
        
        # Add camera discovery configuration if not present
        if not hasattr(config.camera, 'discovery_interval_hours'):
            config.camera.discovery_interval_hours = 1.0  # Default to 1 hour
        
        logger.info(f"✅ Configuration loaded successfully")
        logger.info(f"📝 Log file: {config.log_file}")
        logger.info(f"📡 Camera discovery interval: {str(config.camera.discovery_interval_hours) + 'hours' if config.camera.discovery_interval_hours >= 1 else str(config.camera.discovery_interval_hours) * 60 + 'minutes'} ")
        logger.info(f"🖥️  GUI enabled: {'Yes' if config.enable_gui else 'No'}")
        
        # Initialize dependencies
        logger.info("🔌 Initializing dependencies...")
        dependency_container = initialize_dependencies(config)
        logger.info("✅ Dependencies initialized")
        
        # Create and start application
        app = StreamApplication(config, dependency_container)
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in [signal.SIGTERM, signal.SIGINT]:
            loop.add_signal_handler(sig, app.signal_handler, sig, None)
        
        # Start the application
        await app.start()
        
    except KeyboardInterrupt:
        logger.info("⌨️  Application interrupted by user")
    except Exception as e:
        logger.error(f"💥 Application error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())