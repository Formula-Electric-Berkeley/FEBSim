FROM python:3.12-slim

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy rest of repo
COPY . .

# default command
CMD ["python", "open_loop.py"]