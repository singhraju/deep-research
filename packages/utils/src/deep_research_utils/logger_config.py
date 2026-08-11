"""
Production-ready logging configuration for policy_extractor module.

Features:
- Thread-safe logging for parallel execution
- Rotating file handlers to manage large log files
- Separate loggers for different components with appropriate levels
- Process and thread ID tracking for debugging parallel operations
- Structured logging format for production monitoring
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import threading

# Global lock for thread-safe logger initialization
_logger_lock = threading.Lock()
_initialized_loggers: Dict[str, logging.Logger] = {}

# Default configuration
DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50MB
DEFAULT_BACKUP_COUNT = 10
DEFAULT_FORMAT = (
    '%(asctime)s - PID:%(process)d - TID:%(thread)d - '
    '%(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
DEFAULT_CONSOLE_FORMAT = (
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Environment variable to control console output
ENABLE_CONSOLE_OUTPUT_ENV = "DEEP_RESEARCH_ENABLE_CONSOLE_LOGGING"
CONSOLE_LOG_LEVEL_ENV = "DEEP_RESEARCH_CONSOLE_LOG_LEVEL"


class PolicyExtractorLogger:
    """Centralized logger factory for policy_extractor module."""
    
    def __init__(
        self,
        log_dir: str = DEFAULT_LOG_DIR,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        log_format: str = DEFAULT_FORMAT,
        base_level: int = logging.INFO,
        enable_console_output: Optional[bool] = None,
        console_log_level: Optional[int] = None,
        console_format: str = DEFAULT_CONSOLE_FORMAT,
        console_stream: Optional[Union[sys.stdout.__class__, sys.stderr.__class__]] = None
    ):
        """
        Initialize logging configuration.
        
        Args:
            log_dir: Directory for log files
            max_bytes: Maximum size per log file before rotation
            backup_count: Number of backup files to keep
            log_format: Log message format string
            base_level: Base logging level
            enable_console_output: Whether to enable console output (overrides env var)
            console_log_level: Log level for console output (defaults to WARNING)
            console_format: Format string for console output
            console_stream: Stream for console output (stdout or stderr, defaults to stdout)
        """
        self.log_dir = Path(log_dir)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_format = log_format
        self.base_level = base_level
        
        # Console output configuration
        self.enable_console_output = self._determine_console_output(enable_console_output)
        self.console_log_level = self._determine_console_log_level(console_log_level)
        self.console_format = console_format
        self.console_stream = console_stream or sys.stdout
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup formatters
        self.formatter = logging.Formatter(self.log_format)
        self.console_formatter = logging.Formatter(self.console_format)
        
        # Component-specific log levels
        self.component_levels = {
            'policy_extractor.snowflake_store': logging.WARNING,  # Reduce noise from bulk operations
            'policy_extractor.snowflake_bulk_operations': logging.WARNING,
            'policy_extractor.pipeline': logging.INFO,
            'policy_extractor.api': logging.INFO,
            'policy_extractor.rendering': logging.INFO,
            'policy_extractor.url_extraction': logging.INFO,
            'policy_extractor.chunking': logging.DEBUG,  # More verbose for debugging
        }
    
    def _determine_console_output(self, enable_console_output: Optional[bool]) -> bool:
        """Determine whether console output should be enabled."""
        if enable_console_output is not None:
            return enable_console_output
        
        # Check environment variable via AppConstants
        from deep_research_utils.app_constant import AppConstants
        env_value = AppConstants.DEEP_RESEARCH_ENABLE_CONSOLE_LOGGING.lower()
        return env_value in ("true", "1", "yes", "on")
    
    def _determine_console_log_level(self, console_log_level: Optional[int]) -> int:
        """Determine the log level for console output."""
        if console_log_level is not None:
            return console_log_level
        
        # Check environment variable via AppConstants
        from deep_research_utils.app_constant import AppConstants
        env_level = AppConstants.DEEP_RESEARCH_CONSOLE_LOG_LEVEL.upper()
        return getattr(logging, env_level, logging.WARNING)
    
    def _create_file_handler(self, filename: str, level: int = None) -> logging.handlers.RotatingFileHandler:
        """Create a rotating file handler."""
        log_path = self.log_dir / filename
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        handler.setFormatter(self.formatter)
        if level is not None:
            handler.setLevel(level)
        return handler
    
    def _create_console_handler(self, level: Optional[int] = None) -> logging.StreamHandler:
        """Create console handler with configurable level and stream."""
        if level is None:
            level = self.console_log_level
        
        handler = logging.StreamHandler(self.console_stream)
        handler.setFormatter(self.console_formatter)
        handler.setLevel(level)
        return handler
    
    def get_logger(self, name: str, console_output: Optional[bool] = None) -> logging.Logger:
        """
        Get or create a logger for the specified component.
        
        Args:
            name: Logger name (usually __name__)
            console_output: Whether to also output to console (None uses global setting)
            
        Returns:
            Configured logger instance
        """
        with _logger_lock:
            # Return existing logger if already initialized
            if name in _initialized_loggers:
                return _initialized_loggers[name]
            
            logger = logging.getLogger(name)
            
            # Prevent duplicate handlers if logger already exists
            if logger.handlers:
                logger.handlers.clear()
            
            # Determine log level for this component
            component_level = self.component_levels.get(name, self.base_level)
            logger.setLevel(component_level)
            
            # Main log file handler - all messages
            main_handler = self._create_file_handler('policy_extractor.log')
            logger.addHandler(main_handler)
            
            # Component-specific log file handler
            safe_name = name.replace('.', '_').replace('policy_extractor_', '')
            component_handler = self._create_file_handler(f'{safe_name}.log', component_level)
            logger.addHandler(component_handler)
            
            # Error-only log file handler
            error_handler = self._create_file_handler('errors.log', logging.ERROR)
            logger.addHandler(error_handler)
            
            # Console handler (configurable)
            should_add_console = console_output if console_output is not None else self.enable_console_output
            if should_add_console:
                console_handler = self._create_console_handler()
                logger.addHandler(console_handler)
            
            # Prevent propagation to root logger to avoid duplicate messages
            logger.propagate = False
            
            # Cache the logger
            _initialized_loggers[name] = logger
            
            return logger
    
    def configure_root_logger(self):
        """Configure root logger to capture any unconfigured loggers."""
        root = logging.getLogger()
        
        # Clear existing handlers
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        
        # Add main file handler to root
        root_handler = self._create_file_handler('policy_extractor_root.log')
        root.addHandler(root_handler)
        
        # Add console handler if enabled
        if self.enable_console_output:
            console_handler = self._create_console_handler()
            root.addHandler(console_handler)
        
        root.setLevel(logging.WARNING)
    
    def log_system_info(self):
        """Log system information for debugging."""
        logger = self.get_logger('policy_extractor.system')
        logger.info("=== Policy Extractor Logging Initialized ===")
        logger.info(f"Log directory: {self.log_dir.absolute()}")
        logger.info(f"Max file size: {self.max_bytes / (1024*1024):.1f}MB")
        logger.info(f"Backup count: {self.backup_count}")
        logger.info(f"Console output enabled: {self.enable_console_output}")
        if self.enable_console_output:
            logger.info(f"Console log level: {logging.getLevelName(self.console_log_level)}")
            logger.info(f"Console stream: {self.console_stream.name}")
        logger.info(f"Process ID: {os.getpid()}")
        logger.info("Component log levels:")
        for component, level in self.component_levels.items():
            logger.info(f"  {component}: {logging.getLevelName(level)}")


# Global logger instance
_global_logger_config: Optional[PolicyExtractorLogger] = None


def setup_logging(
    log_dir: str = DEFAULT_LOG_DIR,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    console_output: Optional[bool] = None,
    console_log_level: Optional[int] = None
) -> PolicyExtractorLogger:
    """
    Initialize global logging configuration.
    
    Args:
        log_dir: Directory for log files
        max_bytes: Maximum size per log file before rotation  
        backup_count: Number of backup files to keep
        console_output: Whether to output to console (None uses env var)
        console_log_level: Log level for console output (None uses env var)
        
    Returns:
        PolicyExtractorLogger instance
    """
    global _global_logger_config
    
    if _global_logger_config is None:
        _global_logger_config = PolicyExtractorLogger(
            log_dir=log_dir,
            max_bytes=max_bytes,
            backup_count=backup_count,
            enable_console_output=console_output,
            console_log_level=console_log_level
        )
        _global_logger_config.configure_root_logger()
        _global_logger_config.log_system_info()
    
    return _global_logger_config


def get_logger(name: str, console_output: Optional[bool] = None) -> logging.Logger:
    """
    Get a logger for the specified component.
    
    Args:
        name: Logger name (usually __name__)
        console_output: Whether to also output to console (None uses global setting)
        
    Returns:
        Configured logger instance
    """
    global _global_logger_config
    
    if _global_logger_config is None:
        _global_logger_config = setup_logging()
    
    return _global_logger_config.get_logger(name, console_output)


def cleanup_old_logs(days: int = 30):
    """
    Clean up log files older than specified days.
    
    Args:
        days: Number of days to keep logs
    """
    import time
    from pathlib import Path
    
    logger = get_logger('policy_extractor.cleanup')
    log_dir = Path(DEFAULT_LOG_DIR)
    
    if not log_dir.exists():
        return
    
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    
    for log_file in log_dir.glob('*.log*'):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                logger.info(f"Deleted old log file: {log_file}")
            except Exception as e:
                logger.error(f"Failed to delete log file {log_file}: {e}")


# Context manager for temporary log level changes
class LogLevel:
    """Context manager to temporarily change log level."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
