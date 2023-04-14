# UnstructQA

UnstructQA is a project that utilizes machine learning models to answer user
questions based on guidelines. The project leverages OpenAI and Pinecone to
process and manage the data.

## Requirements

To replicate the process, you will need the following:

1. Clone the repository to your local machine.
2. Create a virtual environment (recommended)
3. API keys for OpenAI and Pinecone, add .env file to root with properties
   `your_openai_api_key` and `your_pinecone_key`
4. In .env file, add `your_openai_api_key` and `your_pinecone_key` with the
   actual API keys.

## Setup Docker

1. Build the Docker image: docker build -t detectron-unstruct .

## Running the Project

1. Run the Docker container with the required API keys: docker run -it --rm
   detectron-unstruct

This will start the container and execute the `app.py` script, guiding you
through the process of vectorizing data and answering user questions based on
pinecone vectorstore data.
