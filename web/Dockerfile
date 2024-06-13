FROM python:3.8-slim

WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir gradio
EXPOSE 7860
ENV GRADIO_SERVER_NAME="127.0.0.1"
ENV GRADIO_ROOT_PATH="/gradio"

CMD ["python", "index.py"]
