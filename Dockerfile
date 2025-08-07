# Use official Python image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY app.py ./
COPY me/ ./me/

# Expose port for Gradio
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
