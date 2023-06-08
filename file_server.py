from http.server import SimpleHTTPRequestHandler, HTTPServer


class CustomHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Content-type', 'text/html; charset=utf-8')
        super().end_headers()


if __name__ == '__main__':
    address = ('', 8099)
    server = HTTPServer(address, CustomHandler)
    server.serve_forever()
