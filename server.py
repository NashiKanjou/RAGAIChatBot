from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import cgi
import logging

from query import *

class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
    def do_HEAD(self):
        self._set_headers()
        
    # GET sends back a Hello world message
    def do_GET(self):
        self.send_response(400)
        #self.wfile.write(json.dumps({'hello': 'world', 'received': 'ok'}))
        
    # POST echoes the message adding a JSON field
    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
        
        # refuse to receive non-json content
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return
            
        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        message = {}
        data = json.loads(self.rfile.read(length))
        
        query = None
        db = None
        model = None
        
        if "query" in data.keys():
            query = data["query"]
        else:
            self.send_response(400)
            self.end_headers()
            return
        
        #TODO: change this to more secure way?
        if "key" in data.keys():
            key = data["key"]
            #change this to use either use username/password to check access or some own API key
            db = key
        if "model" in data.keys():
            model = data["model"]
        self._set_headers()
        
        print("Query: " + query)
        print("Key: " + (db or "default"))
        print("Model: " + (model or "default"))
        
        results = []
        if model == None and db == None:
            results.append(runSingleQuery(query=query))
            #results = runQuery(query=query)
        elif model == None and not db == None:
            results.append(runSingleQuery(query=query, persist_directory=db))
            #results = runQuery(query=query, persist_directory=db)
        elif not model == None and db == None:
            results.append(runSingleQuery(query=query, model=model))
            #results = runQuery(query=query, model=model)
        else:
            results.append(runSingleQuery(query=query, persist_directory=db, model=model))
            #results = runQuery(query=query, persist_directory=db, model=model)
        
        print(results)
        message['response'] = results
        
        # send the results
        self.wfile.write(json.dumps(message).encode('utf-8'))
        
def run(server_class=HTTPServer, handler_class=Server, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    
    print ('Starting httpd on port %d...' % port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')
    
if __name__ == "__main__":
    from sys import argv
    
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()