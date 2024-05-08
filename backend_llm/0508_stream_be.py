from flask import Flask, Response
from threading import Thread
import time

def format_sse(data: str, event=None) -> str:
    msg = f'data: {data}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg

app = Flask(__name__)

temp = ['123','\nasdf','456','\nasdf','789','\nasdf']

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/gen', methods=['POST'])
def gen():
    def test():
        for s in temp:
            time.sleep(1)
            yield s

    return Response(test(), mimetype='text/event-stream')

if __name__=='__main__':
    app.run(debug=True)