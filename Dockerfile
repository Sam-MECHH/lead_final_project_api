FROM continuumio/miniconda3

# Update packages and install nano and curl
RUN apt-get update -y 
RUN apt-get install nano curl -y

# Force Conda to downgrade Python to a stable ML baseline (Python 3.10)
RUN conda install -y python=3.10

# THIS IS SPECIFIC TO HUGGINFACE
# We create a new user named "user" with ID of 1000
RUN useradd -m -u 1000 user
# We switch from "root" (default user when creating an image) to "user" 
USER user
# We set two environmnet variables 
# so that we can give ownership to all files in there afterwards
# we also add /home/user/.local/bin in the $PATH environment variable 
# PATH environment variable sets paths to look for installed binaries
# We update it so that Linux knows where to look for binaries if we were to install them with "user".
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

# We set working directory to $HOME/app (<=> /home/user/app)
WORKDIR $HOME/app

# Copy requirements first to leverage Docker layer caching
COPY --chown=user requirements.txt $HOME/app/requirements.txt

# Install dependencies and clear cache in the same layer to save space
RUN pip install -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . $HOME/app

EXPOSE 7860

# Run FastAPI
CMD ["sh", "-c", "fastapi run src/radio_check_app.py --port ${PORT}"]
