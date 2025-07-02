# config/settings.py
from dataclasses import dataclass
import os
from .database import BaseConfig
from .database import DatabaseConfig
from .camera import CameraConfig
from .detection import DetectionConfig
from .external import ExternalConfig
from .display import DisplayConfig
from .video import VideoConfig
from .ucodec import UcodeSdkConfig
from .rabbitmq import RabbitMQConfig  # NEW
from .redis import RedisConfig        # NEW

@dataclass
class AppConfig(BaseConfig):
    """Main application configuration"""
    # Application settings
    workers_dir: str = "src/workers"
    log_file: str = "src/detections.csv"
    log_level: str = "INFO"
    debug: bool = False
    enable_gui: bool = False
    
    # Component configurations
    database: DatabaseConfig = None
    camera: CameraConfig = None
    detection: DetectionConfig = None
    external: ExternalConfig = None
    display: DisplayConfig = None
    video: VideoConfig = None
    ucode: UcodeSdkConfig = None
    rabbitmq: RabbitMQConfig = None      # NEW
    redis: RedisConfig = None            # NEW
    
    mini_pc_info = None
    
    ENABLE_WEBSOCKET: bool = True
    WEBSOCKET_PORT: int = 8765
    WEBSOCKET_QUALITY: int = 80
    WEBSOCKET_MAX_FPS: int = 15
    
    device_name: str = "minipc-01"
    device_location: str = "unknown"
    api_base_url: str = "https://tabassum.mini-tweet.uz/api/v1"
    api_timeout: int = 30
    api_retry_attempts: int = 3
    api_key: str = ""
    max_worker_threads: int = 4
    
    # Emotion processing mode configuration
    use_rabbitmq: bool = True  # NEW: Enable RabbitMQ or fallback to REST API
    
    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig.from_env()
        if self.camera is None:
            self.camera = CameraConfig.from_env()
        if self.detection is None:
            self.detection = DetectionConfig.from_env()
        if self.external is None:
            self.external = ExternalConfig.from_env()
        if self.display is None:
            self.display = DisplayConfig.from_env()
        if self.video is None:
            self.video = VideoConfig.from_env()
        if self.ucode is None:
            self.ucode = UcodeSdkConfig.from_env()
        if self.rabbitmq is None:
            self.rabbitmq = RabbitMQConfig.from_env()  # NEW
        if self.redis is None:
            self.redis = RedisConfig.from_env()        # NEW
        
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create complete configuration from environment variables"""
        return cls(
            workers_dir=os.getenv('WORKERS_DIR', 'src/workers'),
            log_file=os.getenv('LOG_FILE', 'detections.csv'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            debug=cls.get_env_bool('DEBUG', False),
            api_base_url=os.getenv("API_BASE_URL", "https://tabassum.mini-tweet.uz/api/v1"),
            enable_gui=os.getenv("ENABLE_GUI", True),
            
            # WebSocket settings
            ENABLE_WEBSOCKET=cls.get_env_bool('ENABLE_WEBSOCKET', True),
            WEBSOCKET_PORT=cls.get_env_int('WEBSOCKET_PORT', 8765),
            WEBSOCKET_QUALITY=cls.get_env_int('WEBSOCKET_QUALITY', 80),
            WEBSOCKET_MAX_FPS=cls.get_env_int('WEBSOCKET_MAX_FPS', 15),
            
            # NEW: Emotion processing mode
            use_rabbitmq=cls.get_env_bool('USE_RABBITMQ', True),
            
            # Initialize all component configs
            database=DatabaseConfig.from_env(),
            camera=CameraConfig.from_env(),
            detection=DetectionConfig.from_env(),
            external=ExternalConfig.from_env(),
            display=DisplayConfig.from_env(),
            video=VideoConfig.from_env(),
            ucode=UcodeSdkConfig.from_env(),
            rabbitmq=RabbitMQConfig.from_env(),  # NEW
            redis=RedisConfig.from_env()         # NEW
        )
        
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate required paths exist
        if not os.path.exists(os.path.dirname(self.log_file)):
            errors.append(f"Log directory does not exist: {os.path.dirname(self.log_file)}")
        
        # Validate model files exist
        if not os.path.exists(self.detection.model_prototxt):
            errors.append(f"Model prototxt file not found: {self.detection.model_prototxt}")
        
        if not os.path.exists(self.detection.model_weights):
            errors.append(f"Model weights file not found: {self.detection.model_weights}")
        
        # Validate camera URLs
        if not self.camera.rtsp_urls and not self.camera.fallback_url:
            errors.append("No camera URLs configured")
        
        # Validate database connection
        if not all([self.database.postgres.db_host, self.database.postgres.db_name]):
            errors.append("Database configuration incomplete")
        
        # NEW: Validate RabbitMQ and Redis if enabled
        if self.use_rabbitmq:
            if not self.rabbitmq.validate():
                errors.append("RabbitMQ configuration validation failed")
            
            if not self.redis.validate():
                errors.append("Redis configuration validation failed")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True