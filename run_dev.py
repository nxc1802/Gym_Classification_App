#!/usr/bin/env python3
"""
Development runner for Gym Classification App
Chạy app với debug mode cho development
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import warning suppression FIRST
import utils.suppress_warnings as suppress_warnings

from app import app, video_processor, logger

if __name__ == '__main__':
    logger.info("🔧 Starting app in DEVELOPMENT mode...")
    logger.info("⚠️  Debug mode enabled - app will restart on file changes")
    logger.info("🌐 Development server starting...")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=True)
    except KeyboardInterrupt:
        logger.info("\n🛑 Development server stopped by user")
        # Cleanup if needed
        if video_processor:
            try:
                video_processor.cleanup()
                logger.info("🧹 Cleanup completed")
            except:
                pass 