#!/usr/bin/env python3
import http.server
import socketserver
import os
import sys
import tarfile
import tempfile
import threading
import time
import hashlib
from urllib.parse import quote, unquote
import mimetypes

# Global cache for the compressed archive
archive_cache = {
    'file_path': None,
    'created_time': 0,
    'repo_hash': None
}

def get_repo_hash():
    """Get a hash representing the current state of the repository"""
    hash_md5 = hashlib.md5()
    
    # Walk through all files and get their modification times
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in sorted(files):
            if not file.startswith('.') and not file.endswith('.pyc'):
                filepath = os.path.join(root, file)
                try:
                    mtime = os.path.getmtime(filepath)
                    hash_md5.update(f"{filepath}:{mtime}".encode())
                except OSError:
                    continue
    
    return hash_md5.hexdigest()

def create_cached_archive():
    """Create or reuse cached archive of the repository"""
    global archive_cache
    
    current_hash = get_repo_hash()
    
    # Check if we can reuse existing archive
    if (archive_cache['file_path'] and 
        os.path.exists(archive_cache['file_path']) and 
        archive_cache['repo_hash'] == current_hash):
        print("Reusing existing cached archive")
        return archive_cache['file_path']
    
    # Clean up old archive if it exists
    if archive_cache['file_path'] and os.path.exists(archive_cache['file_path']):
        try:
            os.unlink(archive_cache['file_path'])
            print("Cleaned up old cached archive")
        except OSError:
            pass
    
    print("Creating new cached archive...")
    
    # Create new temporary file for the archive
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz', prefix='auto_drive_')
    temp_file.close()
    
    # Create tar.gz archive
    with tarfile.open(temp_file.name, 'w:gz') as tar:
        # Add all files in the current directory
        for item in os.listdir('.'):
            if (not item.startswith('.') and 
                item != '__pycache__' and
                not item.startswith('auto_drive_')):  # Don't include our own cache files
                print(f"Adding {item} to archive...")
                tar.add(item, arcname=item)
    
    # Update cache
    archive_cache['file_path'] = temp_file.name
    archive_cache['created_time'] = time.time()
    archive_cache['repo_hash'] = current_hash
    
    archive_size = os.path.getsize(temp_file.name)
    print(f"New archive created: {archive_size:,} bytes")
    
    return temp_file.name

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)
    
    def end_headers(self):
        # Add headers to allow cross-origin requests and prevent caching issues
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        if self.path == '/download-all':
            self.handle_download_all()
        elif self.path == '/cache-status':
            self.handle_cache_status()
        else:
            super().do_GET()
    
    def handle_cache_status(self):
        """Return cache status information"""
        global archive_cache
        
        status = {
            'cached': archive_cache['file_path'] is not None and os.path.exists(archive_cache['file_path']),
            'created_time': archive_cache['created_time'],
            'file_size': 0
        }
        
        if status['cached']:
            try:
                status['file_size'] = os.path.getsize(archive_cache['file_path'])
            except OSError:
                status['cached'] = False
        
        response = f"""{{
    "cached": {str(status['cached']).lower()},
    "created_time": {status['created_time']},
    "file_size": {status['file_size']},
    "human_size": "{status['file_size']:,} bytes"
}}"""
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode())
    
    def handle_download_all(self):
        """Create and serve a tar.gz of the entire repository"""
        try:
            # Get or create cached archive
            archive_path = create_cached_archive()
            
            # Get file size
            archive_size = os.path.getsize(archive_path)
            
            # Send the file
            self.send_response(200)
            self.send_header('Content-Type', 'application/gzip')
            self.send_header('Content-Disposition', 'attachment; filename="auto_drive_repository.tar.gz"')
            self.send_header('Content-Length', str(archive_size))
            self.end_headers()
            
            # Stream the file
            with open(archive_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            
            print("Archive download completed")
            
        except Exception as e:
            print(f"Error serving archive: {e}")
            self.send_error(500, f"Error serving archive: {str(e)}")
    
    def list_directory(self, path):
        """Helper to produce a directory listing."""
        try:
            list_dir = os.listdir(path)
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None
        
        list_dir.sort(key=lambda a: a.lower())
        
        # Get relative path for display
        rel_path = os.path.relpath(path, os.getcwd())
        if rel_path == '.':
            rel_path = ''
        
        # Create HTML response
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Auto Drive Repository - {rel_path or 'Root'}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .download-section {{ background-color: #e8f5e8; padding: 20px; border-radius: 5px; margin-bottom: 20px; text-align: center; }}
        .cache-status {{ background-color: #fff3cd; padding: 10px; border-radius: 3px; margin: 10px 0; font-size: 14px; }}
        .download-btn {{ 
            background-color: #4CAF50; 
            color: white; 
            padding: 15px 30px; 
            font-size: 18px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            text-decoration: none;
            display: inline-block;
            margin: 10px;
        }}
        .download-btn:hover {{ background-color: #45a049; }}
        .file-list {{ border-collapse: collapse; width: 100%; }}
        .file-list th, .file-list td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        .file-list th {{ background-color: #f2f2f2; }}
        .directory {{ color: #0066cc; }}
        .file {{ color: #333; }}
        .size {{ text-align: right; }}
    </style>
    <script>
        function updateCacheStatus() {{
            fetch('/cache-status')
                .then(response => response.json())
                .then(data => {{
                    const statusDiv = document.getElementById('cache-status');
                    if (data.cached) {{
                        const date = new Date(data.created_time * 1000).toLocaleString();
                        statusDiv.innerHTML = `📦 Archive cached (created: ${{date}}, size: ${{data.human_size}}) - Downloads will be instant!`;
                        statusDiv.style.backgroundColor = '#d4edda';
                    }} else {{
                        statusDiv.innerHTML = '⏳ No cached archive - first download will create and cache the archive';
                        statusDiv.style.backgroundColor = '#fff3cd';
                    }}
                }})
                .catch(() => {{
                    document.getElementById('cache-status').innerHTML = '❓ Cache status unknown';
                }});
        }}
        
        window.onload = function() {{
            updateCacheStatus();
            setInterval(updateCacheStatus, 5000); // Update every 5 seconds
        }};
    </script>
</head>
<body>
    <div class="header">
        <h1>Auto Drive Repository File Server</h1>
        <p><strong>Current Path:</strong> /{rel_path}</p>
        <p><strong>Total Repository Size:</strong> ~22GB (includes CULane dataset)</p>
        <p>Right-click files to download. Click directories to browse.</p>
    </div>
'''
        
        # Add download all section only for root directory
        if not rel_path:
            html += '''
    <div class="download-section">
        <h2>📦 Download Entire Repository</h2>
        <p>Download the complete auto_drive repository as a single compressed archive</p>
        <div id="cache-status" class="cache-status">Loading cache status...</div>
        <a href="/download-all" class="download-btn">⬇️ Download All (tar.gz)</a>
        <p><small>Archive is cached and reused across multiple downloads for efficiency.<br>
        Archive size will be smaller than 22GB due to compression.</small></p>
    </div>
'''
        
        # Add parent directory link if not in root
        if rel_path:
            parent_path = os.path.dirname(rel_path)
            if parent_path:
                html += f'<p><a href="/{quote(parent_path)}" class="directory">📁 .. (Parent Directory)</a></p>'
            else:
                html += f'<p><a href="/" class="directory">📁 .. (Root)</a></p>'
        
        html += '''
    <table class="file-list">
        <thead>
            <tr>
                <th>Name</th>
                <th>Type</th>
                <th class="size">Size</th>
            </tr>
        </thead>
        <tbody>
'''
        
        for name in list_dir:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            
            # Get file size
            try:
                stat = os.stat(fullname)
                if os.path.isdir(fullname):
                    size = "Directory"
                    file_type = "Directory"
                    css_class = "directory"
                    displayname = f"📁 {name}/"
                    linkname = name + "/"
                else:
                    size = f"{stat.st_size:,} bytes"
                    file_type = "File"
                    css_class = "file"
                    displayname = f"📄 {name}"
            except OSError:
                size = "Unknown"
                file_type = "Unknown"
                css_class = "file"
            
            # Build the link
            if rel_path:
                link_path = f"/{quote(rel_path)}/{quote(linkname)}"
            else:
                link_path = f"/{quote(linkname)}"
            
            html += f'''
            <tr>
                <td><a href="{link_path}" class="{css_class}">{displayname}</a></td>
                <td>{file_type}</td>
                <td class="size">{size}</td>
            </tr>'''
        
        html += '''
        </tbody>
    </table>
    <div style="margin-top: 20px; padding: 10px; background-color: #e8f4f8; border-radius: 5px;">
        <p><strong>Note:</strong> This server provides access to the entire auto_drive repository including:</p>
        <ul>
            <li>Lane detection Python scripts</li>
            <li>Trained models and outputs</li>
            <li>CULane dataset (~22GB)</li>
            <li>Training data and visualizations</li>
        </ul>
    </div>
</body>
</html>'''
        
        encoded = html.encode('utf-8')
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
        return None

def start_file_server(port=8080):
    """Start the file server"""
    handler = CustomHTTPRequestHandler
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Starting file server...")
        print(f"Repository root: {os.getcwd()}")
        print(f"Server running on:")
        print(f"  Local: http://localhost:{port}")
        print(f"  Network: http://192.168.1.181:{port}")
        print(f"\nRepository contains:")
        print(f"  - Python scripts for lane detection")
        print(f"  - CULane dataset (~22GB)")
        print(f"  - Trained models and outputs")
        print(f"\nPress Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number. Using default port 8080.")
    
    start_file_server(port) 