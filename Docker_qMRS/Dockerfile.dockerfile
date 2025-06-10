FROM python:3.10-slim

# Install required dependencies
RUN apt-get update && apt-get install -y \
    xz-utils \
    libx11-6 \
    libxext6 \
    libxft2 \
    libxmu6 \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgfortran5 libstdc++6

# Copy and extract LCModel
COPY lcmodel.xz /opt/lcmodel.xz
WORKDIR /opt
RUN unxz lcmodel.xz && \
    chmod +x lcmodel 

# Add LCModel to the PATH
ENV PATH="/opt:${PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir openpyxl
# Copy the rest of your app
COPY . /app
WORKDIR /app

ENTRYPOINT ["python", "qMRS_pipeline.py"]
