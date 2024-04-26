import requests

response = requests.get("http://192.168.115.38:5010/gen", stream=True)
for line in response.iter_lines():
    decoded_line = line.decode('utf-8')
    print(decoded_line[6:], end='')