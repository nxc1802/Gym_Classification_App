import os
import re
import uuid
import logging
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:vi\/)([0-9A-Za-z_-]{11})',
        r'(?:v\/)([0-9A-Za-z_-]{11})',
        r'(?:\/|%3D)([0-9A-Za-z_-]{11})',
        r'(?:watch.*v=)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def download_youtube_video(url, output_dir):
    """
    Download YouTube video using yt-dlp
    Returns: path to downloaded video file
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp is required for YouTube downloads. Install with: pip install yt-dlp")
    
    # Extract video ID for unique filename
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL - cannot extract video ID")
    
    # Create unique filename
    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"youtube_{video_id}_{unique_id}.%(ext)s"
    output_path = os.path.join(output_dir, output_filename)
    
    # yt-dlp options
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'best[height<=720]/best',  # Prefer 720p or lower for faster processing
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    logger.info(f"Downloading YouTube video: {url}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)
            
            # Check duration limit (10 minutes max for demo)
            if duration > 600:  # 10 minutes
                raise ValueError(f"Video too long ({duration}s). Please use videos shorter than 10 minutes.")
            
            # Download the video
            ydl.download([url])
            
            # Find the actual downloaded file
            base_path = output_path.replace('.%(ext)s', '')
            possible_extensions = ['.mp4', '.mkv', '.webm', '.avi']
            
            downloaded_file = None
            for ext in possible_extensions:
                candidate = base_path + ext
                if os.path.exists(candidate):
                    downloaded_file = candidate
                    break
            
            if not downloaded_file:
                # Fallback: find any file that starts with our pattern
                pattern = os.path.basename(base_path)
                for file in os.listdir(output_dir):
                    if file.startswith(pattern):
                        downloaded_file = os.path.join(output_dir, file)
                        break
            
            if not downloaded_file or not os.path.exists(downloaded_file):
                raise FileNotFoundError("Downloaded file not found")
            
            logger.info(f"YouTube video downloaded: {downloaded_file}")
            return downloaded_file
            
    except Exception as e:
        logger.error(f"YouTube download failed: {str(e)}")
        # Clean up any partial downloads
        try:
            base_path = output_path.replace('.%(ext)s', '')
            for ext in ['.mp4', '.mkv', '.webm', '.avi', '.part']:
                candidate = base_path + ext
                if os.path.exists(candidate):
                    os.remove(candidate)
        except:
            pass
        raise

def get_video_info(url):
    """Get basic info about YouTube video without downloading"""
    try:
        import yt_dlp
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0)
            }
    except Exception as e:
        logger.error(f"Failed to get video info: {str(e)}")
        return None 