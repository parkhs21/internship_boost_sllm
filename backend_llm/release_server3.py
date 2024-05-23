from flask import Flask, request, jsonify, Response
from flask_basicauth import BasicAuth
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from datetime import datetime, timedelta
from threading import Thread
import torch
import gc
from setproctitle import setproctitle
import subprocess
from threading import Timer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
setproctitle('boost_llm:5009')

app.config['BASIC_AUTH_USERNAME'] = 'username'
app.config['BASIC_AUTH_PASSWORD'] = 'password123!@#'
basic_auth = BasicAuth(app)

MODEL_LIST = [
    "codellama/CodeLlama-7b-Instruct-hf",
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "unsloth/llama-3-70b-Instruct-bnb-4bit"
]

MODEL_ID_CONVERT = {
    "A6000": 0,
    "A5500": 1,
    "A5000": 2,
    "A4000": 3,
}

# when the model last used
last_used_times = {}

## Model cached for re-use
model_loaded = {}

model_name = "jhgan/ko-sroberta-nli"
model_kwargs = {'device': 'cuda:2'}
encode_kwargs = {'normalize_embeddings': True}
hf_emb = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
vectorstore = FAISS.load_local('./korean_civil_act_ko-sroberta-nli', hf_emb, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'lambda_mult': 0.15}
)

def get_loaded_model_list():
    global model_loaded
    if not model_loaded: return [("Nothing loaded\n", None)]
    
    loaded_list = []
    for model_id in model_loaded.keys():
        gpu_idx = model_loaded[model_id]["config"]["gpu_index"]
        loaded_list.append((model_id+'\n', f"GPU {gpu_idx}"))
    return loaded_list

def unload_model(model_id):
    ## Unload a model and tokenizer from memory.
    global model_loaded
    
    del model_loaded[model_id]["model"]
    del model_loaded[model_id]["tokenizer"]
    del model_loaded[model_id]
    
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_id: str, quantize: bool, bits: int, gpu: str):
    ## Load a model into the cache with optional quantization.
    quantization_config = BitsAndBytesConfig(bits=bits) if quantize else None
    
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map=f'cuda:{MODEL_ID_CONVERT[gpu]}')
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map=f'cuda:{MODEL_ID_CONVERT[gpu]}')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    return model, tokenizer

# 주기적으로 모델의 상태를 확인하고 일정 시간 동안 사용되지 않은 모델을 언로드
def unload_unused_models():
    global last_used_times, model_loaded
    
    # 언로드할 모델을 저장할 리스트
    models_to_unload = []
    
    # 현재 시간
    current_time = datetime.now()
    
    # 일정 시간(예: 30분) 동안 사용되지 않은 모델을 찾아서 리스트에 추가
    for model_id, last_used_time in last_used_times.items():
        if (current_time - last_used_time) > timedelta(minutes=30):  # 30분
            models_to_unload.append(model_id)
    
    # 모델 언로드
    for model_id in models_to_unload:
        unload_model(model_id)
        del last_used_times[model_id]  # 마지막 사용 시간 딕셔너리에서 제거
    
    # 주기적으로 반복되도록 타이머 설정
    # Timer(1800, unload_unused_models).start()  # 30분(1800초)마다 실행

# 언로드 함수 최초 실행
unload_unused_models()

def format_sse(data: str, event=None) -> str:
    msg = f'data: {data}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg

# Example api
@app.route('/health', methods=['GET'])
@basic_auth.required
def health_check():
    response = {
        "isSuccess": True,
        "result": "healty!"
    }
    return jsonify(response)


# API NO.1 - GPU에 현재 로드할 수 있는 model list 반환
@app.route('/model/list', methods=['GET'])
@basic_auth.required
def model_list():
    global MODEL_LIST
    
    args = request.args
    
    response = {
        "isSuccess": False,
        "result": ""
    }
    
    response["isSuccess"] = True
    response["result"] = MODEL_LIST
    return jsonify(response)


# API NO.2 - GPU-index와 model을 선택하여 해당 gpu에 모델을 load
@app.route('/model/load', methods=['POST'])
@basic_auth.required
def model_load():
    global model_loaded
    
    data = request.json
    model_id = data.get('model')
    gpu_idx = data.get('gpu_index')
    quantize = data.get('quantize', True)
    bits = data.get('bits', 4)
    
    data["quantize"] = quantize
    data["bits"] = bits
    
    response = {
        "isSuccess": False,
        "result": ""
    }
    
    if model_id in model_loaded:
        response["result"] = f"{model_id} 모델은 이미 로딩 중입니다."
    else:
        try:
            model, tokenizer = load_model(model_id, quantize, bits, gpu_idx)
            model_loaded[model_id] = {"model": model, "tokenizer": tokenizer, "config": data}
            
            gpu_info = get_gpu_info()['result']
            loaded_model_list = get_loaded_model_list()
            
            response["isSuccess"] = True
            response["result"] = {
                "gpu_info": gpu_info,
                "model_loaded_list": loaded_model_list
            }
        except Exception as e:
            #response["result"] = "모델 로드에 실패했습니다. 다시 시도 해주세요."
            response["result"] = f"Model load failed. Error: {str(e)}"
    return jsonify(response)
    
    
# API NO.3 - 선택한 model을 unload
@app.route('/model/unload', methods=['POST'])
@basic_auth.required
def model_unload():
    global model_loaded
    
    data = request.json
    model_id = data.get('model')
    
    response = {
        "isSuccess": False,
        "result": ""
    }
    
    if not model_id in model_loaded:
        response["result"] = "해당 모델은 로딩되어 있지 않습니다."
    else:
        try:
            unload_model(model_id)
            
            response["isSuccess"] = True
            response["result"] = get_loaded_model_list()
        except:
            response["result"] = "모델 언로드에 실패했습니다. 다시 시도 해주세요."

    return jsonify(response)


# API NO.4 - 현재 gpu에 load된 모델 목록 조회
@app.route('/model/loaded-list', methods=['GET'])
@basic_auth.required
def model_loaded_list():
    global model_loaded
    
    args = request.args
    
    response = {
        "isSuccess": False,
        "result": ""
    }
   
    response["isSuccess"] = True
    response["result"] = get_loaded_model_list()
    return jsonify(response)


# API NO.5 - input을 받아 모델을 거쳐 output으로 반환 (기존코드)
@app.route('/generate-text', methods=['POST'])
@basic_auth.required
def generate_text():
    global model_loaded

    start = datetime.now()
    
    data = request.json
    model_id = data.get('model')
    input_text = data.get('prompt')
    max_new_token = data.get('max_new_token')
    
    response = {
        "isSuccess": False,
        "result": ""
    }
    
    if not model_id in model_loaded:
        response["result"] = "해당 모델은 로딩되어 있지 않습니다. 로드 후, 다시 시도 해주세요."
        return jsonify(response)
    
    model = model_loaded[model_id]["model"]
    tokenizer = model_loaded[model_id]["tokenizer"]
    config = model_loaded[model_id]["config"]
    
    try:
        input_tokens = tokenizer(input_text, return_tensors="pt")
        output = model.generate(**input_tokens, max_new_tokens=max_new_token)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        response["isSuccess"] = True
        
        config["input_text"] = input_text
        config["output_text"] = output_text
        config["max_new_token"] = max_new_token
        
        end = datetime.now()
        config["time"] = (end-start).total_seconds()
        
        response["result"] = config

        # 모델을 사용한 시간 갱신
        last_used_times[model_id] = datetime.now()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        response["result"] = "메모리 부족으로 텍스트 생성에 실패했습니다. 나중에 다시 시도해주세요."
    except:
        response["result"] = "답변 생성에 실패했습니다. 다시 시도 해주세요."
    
    return jsonify(response)

# API NO.5.1 by cjh - input을 받아 모델을 거쳐 output으로 반환
@app.route('/generate', methods=['POST'])
@basic_auth.required
def generate_text2():
    global model_loaded

    start = datetime.now()
    
    data = request.json
    model_id = data.get('model')
    input_text = data.get('prompt')
    max_new_token = data.get('max_new_token')
    do_sample = data.get('do_sample', True)
    temperature = data.get('temperature', 0.9)
    top_p = data.get('top_p', 0.7)
    
    response = {
        "isSuccess": False,
        "result": ""
    }
    
    if not model_id in model_loaded:
        response["result"] = "해당 모델은 로딩되어 있지 않습니다. 로드 후, 다시 시도 해주세요."
        return jsonify(response)
    
    model = model_loaded[model_id]["model"]
    tokenizer = model_loaded[model_id]["tokenizer"]
    config = model_loaded[model_id]["config"]
    
    try:
                
        prompt_refactor_front =  """ [[[2]{"role": "system", "content": "Your role is to provide concise and accurate answers based on the inputs you receive. You have to cover a wide range of topics, from programming to general knowledge questions. Summarize your points first, and deliver the necessary information concisely. Make sure you have answered the questions clearly related to programming. Do not repeat the questions in your answer, but provide them in a structured and clear format. Answer only once per question.You should not create your own "role": "system", "content"."
        }]],

            [[[1]{"role": "user", "content": " """ 
        prompt_refactor_end = """"
        }]],
            [[[0]{"role": "system", "content": "
            
        """
        prompts = prompt_refactor_front+input_text+prompt_refactor_end
        input_tokens = tokenizer(prompts, return_tensors="pt")
        
        input_tokens = input_tokens.to('cuda')
        if do_sample == True:
            output = model.generate(**input_tokens, max_new_tokens=max_new_token,do_sample=do_sample, temperature=temperature ,top_p=top_p)
        else:
            output = model.generate(**input_tokens, max_new_tokens=max_new_token,do_sample=do_sample)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        start_index = output_text.find('[[[0]{"role": "system", "content": "')+len('[[[]]0]{"role": "system", "content": "')
        end_index = output_text.find('}]]', start_index)   # '}' 뒤에 오는 '"' 포함
        # 추출된 텍스트 출력
        if start_index != -1 and end_index != -1 and (end_index >start_index ) :
           output_text = output_text[start_index:end_index-1]
        else:
           output_text = output_text[start_index:-1]
        response["isSuccess"] = True
        
        config["input_text"] = input_text
        config["output_text"] = output_text
        config["max_new_token"] = max_new_token
        
        end = datetime.now()
        config["time"] = (end-start).total_seconds()
        
        response["result"] = config

        # 모델을 사용한 시간 갱신
        last_used_times[model_id] = datetime.now()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        response["result"] = "메모리 부족으로 텍스트 생성에 실패했습니다. 나중에 다시 시도해주세요."
    except:
        response["result"] = "답변 생성에 실패했습니다. 다시 시도 해주세요."
    
    return jsonify(response)

# API NO.5.2 by rjh - RefactorGptService
@app.route('/generate-refactorcode', methods=['POST'])
@basic_auth.required
def generate_text3():
    global model_loaded

    start = datetime.now()
    
    data = request.json
    model_id = data.get('model')
    input_text = data.get('prompt')
    max_new_token = data.get('max_new_token')
    do_sample = data.get('do_sample', True)
    temperature = data.get('temperature', 0.9)
    top_p = data.get('top_p', 0.7)
    
    response = {
        "isSuccess": False,
        "result": ""
    }
    
    if not model_id in model_loaded:
        response["result"] = "해당 모델은 로딩되어 있지 않습니다. 로드 후, 다시 시도 해주세요."
        return jsonify(response)
    
    model = model_loaded[model_id]["model"]
    tokenizer = model_loaded[model_id]["tokenizer"]
    config = model_loaded[model_id]["config"]
    
    try:
                
        prompt_refactor_front =  """ [[[2]{"role": "system", "content": "Your role is to provide concise and accurate answers based on the inputs you receive. You have to cover a wide range of topics, from programming to general knowledge questions. Summarize your points first, and deliver the necessary information concisely. Make sure you have answered the questions clearly related to programming. Do not repeat the questions in your answer, but provide them in a structured and clear format. Answer only once per question.You should not create your own "role": "system", "content"."
        }]],

            [[[1]{"role": "user", "content": " """ 
        prompt_refactor_end = """"
        }]],
            [[[0]{"role": "system", "content": "
            
        """
        prompts = prompt_refactor_front+input_text+prompt_refactor_end
        print(prompts)
        input_tokens = tokenizer(prompts, return_tensors="pt")
        
        input_tokens = input_tokens.to('cuda')
        if do_sample == True:
            output = model.generate(**input_tokens, max_new_tokens=max_new_token,do_sample=do_sample, temperature=temperature ,top_p=top_p)
        else:
            output = model.generate(**input_tokens, max_new_tokens=max_new_token,do_sample=do_sample)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        start_index = output_text.find('[[[0]{"role": "system", "content": "')+len('[[[]]0]{"role": "system", "content": "')
        end_index = output_text.find('}]]', start_index)   # '}' 뒤에 오는 '"' 포함
        # 추출된 텍스트 출력
        if start_index != -1 and end_index != -1 and (end_index >start_index ) :
           output_text = output_text[start_index:end_index-1]
        else:
           output_text = output_text[start_index:-1]
        response["isSuccess"] = True
        
        config["input_text"] = input_text
        config["output_text"] = output_text
        config["max_new_token"] = max_new_token
        
        end = datetime.now()
        config["time"] = (end-start).total_seconds()
        
        response["result"] = config

        # 모델을 사용한 시간 갱신
        last_used_times[model_id] = datetime.now()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        response["result"] = "메모리 부족으로 텍스트 생성에 실패했습니다. 나중에 다시 시도해주세요."
    except:
        response["result"] = "답변 생성에 실패했습니다. 다시 시도 해주세요."
    
    return jsonify(response)

# API NO.5.2 by rjh - JavaDocService
@app.route('/generate-javadoc', methods=['POST'])
@basic_auth.required
def generate_text4():
    global model_loaded

    start = datetime.now()
    
    data = request.json
    model_id = data.get('model')
    input_text = data.get('prompt')
    max_new_token = data.get('max_new_token')
    do_sample = data.get('do_sample', True)
    temperature = data.get('temperature', 0.9)
    top_p = data.get('top_p', 0.7)
    
    response = {
        "isSuccess": False,
        "result": ""
    }
    
    if not model_id in model_loaded:
        response["result"] = "해당 모델은 로딩되어 있지 않습니다. 로드 후, 다시 시도 해주세요."
        return jsonify(response)
    
    model = model_loaded[model_id]["model"]
    tokenizer = model_loaded[model_id]["tokenizer"]
    config = model_loaded[model_id]["config"]
    
    try:
                
        prompt_refactor_front =  """ [[[2]{"role": "system", "content": "Your role is to provide concise and accurate answers based on the inputs you receive. You have to cover a wide range of topics, from programming to general knowledge questions. Summarize your points first, and deliver the necessary information concisely. Make sure you have answered the questions clearly related to programming. Do not repeat the questions in your answer, but provide them in a structured and clear format. Answer only once per question.You should not create your own "role": "system", "content"."
        }]],

            [[[1]{"role": "user", "content": " """ 
        prompt_refactor_end = """"
        }]],
            [[[0]{"role": "system", "content": "
            
        """
        prompts = prompt_refactor_front+input_text+prompt_refactor_end
        print(prompts)
        input_tokens = tokenizer(prompts, return_tensors="pt")
        
        input_tokens = input_tokens.to('cuda')
        if do_sample == True:
            output = model.generate(**input_tokens, max_new_tokens=max_new_token,do_sample=do_sample, temperature=temperature ,top_p=top_p)
        else:
            output = model.generate(**input_tokens, max_new_tokens=max_new_token,do_sample=do_sample)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        start_index = output_text.find('[[[0]{"role": "system", "content": "')+len('[[[]]0]{"role": "system", "content": "')
        end_index = output_text.find('}]]', start_index)   # '}' 뒤에 오는 '"' 포함
        # 추출된 텍스트 출력
        if start_index != -1 and end_index != -1 and (end_index >start_index ) :
           output_text = output_text[start_index:end_index-1]
        else:
           output_text = output_text[start_index:-1]
        response["isSuccess"] = True
        
        config["input_text"] = input_text
        config["output_text"] = output_text
        config["max_new_token"] = max_new_token
        
        end = datetime.now()
        config["time"] = (end-start).total_seconds()
        
        response["result"] = config

        # 모델을 사용한 시간 갱신
        last_used_times[model_id] = datetime.now()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        response["result"] = "메모리 부족으로 텍스트 생성에 실패했습니다. 나중에 다시 시도해주세요."
    except:
        response["result"] = "답변 생성에 실패했습니다. 다시 시도 해주세요."
    
    return jsonify(response)

# API NO.6 - 현재 gpu상태 반환
@app.route('/gpu-info', methods=['GET'])
def get_gpu_info():
    try:
        cmd_output = subprocess.check_output(['nvidia-smi', '--format=csv,nounits', '--query-gpu=name,memory.used,memory.total'])
        cmd_output = cmd_output.decode('utf-8')
        
        response = {
            "isSuccess": False,
            "result": ""
        }
        
        infos = []
        lines = cmd_output.strip().split('\n')
        headers = lines[0].split(', ')
        for line in lines[1:]:
            values = line.split(', ')
            infos.append({ headers[i]: values[i] for i in range(len(headers)) })
            
        response["isSuccess"] = True
        response["result"] = infos
        # Return the dictionary directly, Flask will automatically convert it to JSON
        return response
    except subprocess.CalledProcessError as e:
        return {'error': f'Failed to retrieve GPU information: {e.output.decode("utf-8")}'}
    except Exception as e:
        return {'error': f'An error occurred: {str(e)}'}
    
# API NO.7 - 현재 GPU와 로드된 모델 정보를 반환
@app.route('/model/loaded', methods=['GET'])
@basic_auth.required
def model_gpu_info():
    # Get loaded model list
    loaded_models = get_loaded_model_list()

    # Get GPU information
    gpu_information = get_gpu_info()

    # Prepare response
    response = {
        "isSuccess": True,
        "result": {
            "model_loaded_list": loaded_models,
            "gpu_info": gpu_information['result'] if gpu_information["isSuccess"] else []
        }
    }

    return jsonify(response)


@app.route('/generate/stream', methods=['POST'])
@basic_auth.required
def generate_stream():
    global model_loaded
    
    data = request.json
    model_id = data.get('model')
    input_text = data.get('prompt')
    max_new_token = data.get('max_new_token')
    do_sample = data.get('do_sample', True)
    temperature = data.get('temperature', 0.9)
    top_p = data.get('top_p', 0.7)
    rag = data.get('rag', None)
    print(rag)
    
    response = {
        "isSuccess": False,
        "result": ""
    }
    
    if not model_id in model_loaded:
        response["result"] = "해당 모델은 로딩되어 있지 않습니다. 로드 후, 다시 시도 해주세요."
        return jsonify(response)
    
    model = model_loaded[model_id]["model"]
    tokenizer = model_loaded[model_id]["tokenizer"]
    config = model_loaded[model_id]["config"]
    
    prompt = lambda input_text: f"""[4]{{"role": "system", "content": <sos>What is my role?
    <eos>}},
    [3]{{"role": "user", "content": <sos>Your role is to provide concise and accurate answers based on the inputs you receive. You have to cover a wide range of topics, from programming to general knowledge questions. Summarize your points first, and deliver the necessary information concisely. Do not repeat the questions in your answer, but provide them in a structured and clear format. Answer only once per question.You should not create your own "role": "system", "content".
    <eos>}},
    [2]{{"role": "system", "content": <sos>Okey. what can I do for  you?
    <eos>}},
    [1]{{"role": "user", "content": <sos>{input_text}
    <eos>}},
    [0]{{"role": "system", "content": <sos>
    """
    
    if rag is not None:
        context = retriever.invoke(input_text)
        prompt = lambda input_text: f'''You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {input_text}
        Context: {context}
        Answer:'''
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    inputs = tokenizer(prompt(input_text), return_tensors="pt").to("cuda:0")
    
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_token, do_sample=False)
    if do_sample:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    def output():
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        for s in streamer:
            if '<eos>' in s:
                end_index = str(s).index('<eos>')
                yield s[:end_index]
                break
            yield s
            
    return Response(output(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='192.168.115.38', port=5009)