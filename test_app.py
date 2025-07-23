#!/usr/bin/env python3
"""
Test script để kiểm tra app có thể chạy được không
Chỉ test imports và cấu hình cơ bản, không load models
"""

import sys
import os

def test_imports():
    """Test tất cả imports cần thiết"""
    try:
        print("🔍 Testing imports...")
        
        # Test Flask imports
        from flask import Flask, request, render_template, jsonify
        print("✅ Flask imports OK")
        
        # Test utils imports
        import utils.suppress_warnings as suppress_warnings
        print("✅ Utils imports OK")
        
        # Test basic app creation
        app = Flask(__name__)
        print("✅ Flask app creation OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_directories():
    """Test các thư mục cần thiết có tồn tại không"""
    required_dirs = [
        'static',
        'static/css',
        'static/js',
        'static/uploads',
        'templates',
        'models',
        'utils'
    ]
    
    print("🔍 Testing directories...")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            return False
    return True

def main():
    print("🚀 Testing Gym Classification App...")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    print()
    
    # Test directories  
    if not test_directories():
        success = False
    
    print()
    print("=" * 50)
    
    if success:
        print("✅ All tests passed! App should be ready to run.")
        print("📝 Note: Model weights are needed for full functionality.")
        print("💡 Use 'python app.py' to start the full application.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 