# Linux container. Stereo path only: the Apple Spatial Audio decoder needs
# macOS and AVFoundation, so it is deliberately absent here.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY cli/ ./cli/

# HRTF data, fetched at build time rather than committed to the repo
RUN curl -fL --retry 3 -o app/kemar.sofa \
      https://sofacoustics.org/data/database/mit/mit_kemar_normal_pinna.sofa

RUN mkdir -p app/jobs
ENV PORT=8765
EXPOSE 8765
CMD ["sh","-c","uvicorn server:app --app-dir /app/app --host 0.0.0.0 --port ${PORT}"]
