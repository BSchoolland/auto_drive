import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class GraphHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, loss_data_ref=None, **kwargs):
        self.loss_data_ref = loss_data_ref
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        # Suppress request logging
        pass
    
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Live Loss Graph</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    .chart-container {
                        width: 800px;
                        height: 400px;
                        margin: 20px;
                    }
                    .charts-wrapper {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }
                </style>
            </head>
            <body>
                <h1>Live Training Loss</h1>
                <div class="charts-wrapper">
                    <div class="chart-container">
                        <h3>All Training History</h3>
                        <canvas id="fullChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <h3>Recent Progress (Last 10%)</h3>
                        <canvas id="recentChart"></canvas>
                    </div>
                </div>
                <script>
                    const fullCtx = document.getElementById('fullChart').getContext('2d');
                    const recentCtx = document.getElementById('recentChart').getContext('2d');
                    
                    const chartConfig = {
                        type: 'scatter',
                        data: {
                            datasets: [{
                                label: 'Loss',
                                data: [],
                                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                                borderColor: 'rgb(75, 192, 192)',
                                pointRadius: 2,
                                showLine: true,
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    type: 'linear',
                                    position: 'bottom',
                                    title: {
                                        display: true,
                                        text: 'Epoch'
                                    }
                                },
                                y: {
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: 'Loss'
                                    }
                                }
                            },
                            plugins: {
                                legend: {
                                    display: true
                                }
                            }
                        }
                    };
                    
                    const fullChart = new Chart(fullCtx, JSON.parse(JSON.stringify(chartConfig)));
                    const recentChart = new Chart(recentCtx, JSON.parse(JSON.stringify(chartConfig)));
                    
                    function updateCharts() {
                        fetch('/data')
                            .then(response => response.json())
                            .then(data => {
                                if (data.length === 0) return;
                                
                                // Full chart - all data
                                const fullData = data.map(d => ({x: d.epoch, y: d.loss}));
                                fullChart.data.datasets[0].data = fullData;
                                fullChart.update('none');
                                
                                // Recent chart - last 10% of max epochs
                                const maxEpoch = Math.max(...data.map(d => d.epoch));
                                const tenPercentThreshold = Math.max(0, maxEpoch - Math.floor(maxEpoch * 0.1));
                                const recentData = data
                                    .filter(d => d.epoch >= tenPercentThreshold)
                                    .map(d => ({x: d.epoch, y: d.loss}));
                                
                                recentChart.data.datasets[0].data = recentData;
                                recentChart.update('none');
                            });
                    }
                    
                    setInterval(updateCharts, 500);
                    updateCharts();
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        
        elif self.path == "/data":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            # Serve live data from memory, not from file
            self.wfile.write(json.dumps(self.loss_data_ref).encode())
        
        else:
            self.send_response(404)
            self.end_headers()

def start_server(loss_data_ref, port=8000):
    def handler(*args, **kwargs):
        return GraphHandler(*args, loss_data_ref=loss_data_ref, **kwargs)
    
    server = HTTPServer(("localhost", port), handler)
    print(f"Server running at http://localhost:{port}")
    server.serve_forever()

def start_server_thread(loss_data_ref, port=8000):
    server_thread = threading.Thread(target=start_server, args=(loss_data_ref, port), daemon=True)
    server_thread.start()
    return server_thread 