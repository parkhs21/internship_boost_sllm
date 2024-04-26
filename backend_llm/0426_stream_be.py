from flask import Flask, Response
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from threading import Thread
from transformers import TextIteratorStreamer

quantization_config = BitsAndBytesConfig()
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", quantization_config=quantization_config, device_map='cuda:1')
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", device_map='cuda:1')

streamer = TextIteratorStreamer(tokenizer, skip_prompt=False)
inputs = tokenizer(["An increasing sequence: one,"], return_tensors="pt").to("cuda:1")
kwargs = dict(input_ids=inputs["input_ids"], streamer=streamer, max_new_tokens=50)
                
def format_sse(data: str, event=None) -> str:
    msg = f'data: {data}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/genp', methods=['GET'])
def gnep():
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    for s in streamer:
        print(s, end='')
    return {}

@app.route('/gen', methods=['GET'])
def gen():
    def test():
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        for s in streamer:
            yield format_sse(s)

    return Response(test(), mimetype='text/event-stream')

if __name__=='__main__':
    app.run(debug=True, host='192.168.115.38', port=5010)