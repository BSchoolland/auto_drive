#!/usr/bin/env python3
import http.server
import socketserver
import os
import tarfile
import tempfile

# Create the archive once when server starts
print("Creating archive of repository...")
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz')
temp_file.close()

with tarfile.open(temp_file.name, 'w:gz') as tar:
    for item in os.listdir('.'):
        if not item.startswith('.') and item != '__pycache__':
            print(f"Adding {item}...")
            tar.add(item, arcname=item)

archive_size = os.path.getsize(temp_file.name)
print(f"Archive created: {archive_size:,} bytes")
print(f"Archive saved as: {temp_file.name}")

class DownloadHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve the archive for any request
        self.send_response(200)
        self.send_header('Content-Type', 'application/gzip')
        self.send_header('Content-Disposition', 'attachment; filename="auto_drive_repository.tar.gz"')
        self.send_header('Content-Length', str(archive_size))
        self.end_headers()
        
        with open(temp_file.name, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
        
        print("Download served")

if __name__ == "__main__":
    port = 8080
    with socketserver.TCPServer(("", port), DownloadHandler) as httpd:
        print(f"Download server running on:")
        print(f"  Local: http://localhost:{port}")
        print(f"  Network: http://192.168.1.181:{port}")
        print("Any access will immediately download the archive")
        httpd.serve_forever() 