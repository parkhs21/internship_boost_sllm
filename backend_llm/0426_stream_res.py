import requests

session = requests.Session()
session.auth = ('username', 'password123!@#')

body = {
    "model": "codellama/CodeLlama-7b-Instruct-hf",
    "prompt": "def temp(a,b):\n    c = a+b\n    print(c)",
    "max_new_token": 50
}

response = session.post("http://192.168.115.38:5009/generate/stream", json=body, stream=True)
for line in response.iter_lines():
    # print(line)
    decoded_line = line.decode('utf-8')
    print(decoded_line)
    print("ASDF")
    # if decoded_line[6:]=="\n\n": print("ASDF")
    # print(decoded_line[6:])