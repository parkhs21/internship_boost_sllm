import requests

response = requests.post("http://127.0.0.1:5000/gen", stream=True)
for line in response.iter_content():
    print(line)
    # decoded_line = line.decode('utf-8')
    # print(decoded_line)
    # if decoded_line[6:]=="\n\n": print("ASDF")
    # print(decoded_line[6:])