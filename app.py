# Import warning suppression FIRST
import utils.suppress_warnings as suppress_warnings

from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import time
import uuid
from werkzeug.utils import secure_filename
import logging

from models.video_processor import VideoProcessor
from utils.youtube_downloader import download_youtube_video

# Configure clean logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize video processor eagerly at startup
logger.info("🚀 Initializing Gym Demo App...")
logger.info("📦 Loading models and setting up components...")
logger.info("⏳ This may take a few moments on first startup...")

try:
    video_processor = VideoProcessor()
    logger.info("✅ All models and components loaded successfully!")
    logger.info("🎯 App ready to process videos!")
except Exception as e:
    logger.error(f"❌ Error during startup initialization: {e}")
    logger.info("⚠️  App will continue in demo mode")
    video_processor = None

# Supported video formats
ALLOWED_EXTENSIONS = {
    'mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv', 'flv', 'm4v', '3gp'
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_youtube_url(url):
    """Check if URL is a YouTube video"""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    return any(domain in url.lower() for domain in youtube_domains)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload or YouTube URL"""
    global video_processor
    
    # Check if video processor is available
    if video_processor is None:
        logger.error("Video processor not initialized")
        return jsonify({'error': 'Server not ready. Please restart the application.'}), 503
    
    try:
        video_path = None
        
        # Check if it's a YouTube URL
        youtube_url = request.form.get('youtube_url', '').strip()
        if youtube_url:
            if not is_youtube_url(youtube_url):
                return jsonify({'error': 'Invalid YouTube URL'}), 400
            
            logger.info(f"Processing YouTube URL: {youtube_url}")
            try:
                video_path = download_youtube_video(youtube_url, app.config['UPLOAD_FOLDER'])
            except Exception as e:
                logger.error(f"YouTube download error: {str(e)}")
                return jsonify({'error': f'Failed to download YouTube video: {str(e)}'}), 400
        
        # Check if it's a file upload
        elif 'video' in request.files:
            file = request.files['video']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and allowed_file(file.filename):
                # Generate unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(video_path)
                logger.info(f"File uploaded: {video_path}")
            else:
                return jsonify({'error': 'Invalid file format'}), 400
        else:
            return jsonify({'error': 'No video file or YouTube URL provided'}), 400
        
        # Process video
        logger.info(f"Starting video processing: {video_path}")
        start_time = time.time()
        
        results = video_processor.process_video(video_path)
        
        total_time = time.time() - start_time
        results['total_processing_time'] = round(total_time, 2)
        
        logger.info(f"Video processing completed in {total_time:.2f}s")
        
        # Clean up uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        # Clean up on error
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Run startup check first
    logger.info("🔍 Running startup verification...")
    try:
        from startup_check import run_startup_check
        run_startup_check()
    except ImportError:
        logger.info("Startup check not available, proceeding...")
    
    # Start the Flask app
    logger.info("🌐 Starting Flask server in PRODUCTION mode...")
    logger.info("📱 App will be available at: http://localhost:5000")
    logger.info("📱 Network access at: http://0.0.0.0:5000")
    logger.info("🔧 For development mode, use: python run_dev.py")
    logger.info("🛑 Press Ctrl+C to stop")
    logger.info("=" * 50)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
        # Cleanup if needed
        if video_processor:
            try:
                video_processor.cleanup()
                logger.info("🧹 Cleanup completed")
            except:
                pass 