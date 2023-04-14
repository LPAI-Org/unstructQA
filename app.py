import glob
import json
import tiktoken
from tqdm.auto import tqdm

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.text import partition_text
import time
import openai
import pinecone
import os
from dotenv import load_dotenv

# system message to 'prime' the model
primer = f"""You are a professional Q&A manager that knows mortgage guidelines and requirements. A brilliant system that answers
user questions based on the information provided by the user above
each question. When you find the answer, return only the response to that is directly for the question alone. Do not verbose.

If the information can not be found in the information
provided by the user, you truthfully say, "I don't know".

The following acronyms can be used interchangeably:

-CLTV = Combined Loan To Value
-LTV = Loan To Value
-Fico = Credit Score
-Credit = Credit Score
-Score = Credit Score
-PIW = Property Inspection Waiver
-DTI = Debt to income or Debt to income ratio
-Requirement = Guidelines

If someone uses the acronyms above, the name after the equal sign is its terminology.

"""

load_dotenv()
openai.api_key= os.environ.get('OPENAI_API_KEY')
pinecone_key = os.getenv('PINECONE_KEY')


#initialize the embedding with Ada

embed_model = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)
    
# Time to initialize Pinecone store for our embeddings
index_name = 'lenders'

# initialize connection to pinecone
pinecone.init(
    api_key=pinecone_key,  # app.pinecone.io (console)
    environment="us-east-1-aws"  # next to API key in console
)

# check if index already exists
index_exists = index_name in pinecone.list_indexes()

if not index_exists:
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine'
    )

# connect to index
index = pinecone.GRPCIndex(index_name)
# view index stats
print(f'Current Pinecone Vector Status: ', index.describe_index_stats())

# Check if the user wants to proceed with vectorizing new data
if index_exists:
    user_input = input("Pinecone index already exists. Do you want to proceed with vectorizing new data? (yes/no): ")
    proceed_with_vectorizing = user_input.lower() == 'yes'
else:
    proceed_with_vectorizing = True

# ...

if proceed_with_vectorizing:
    # Now we will upsert all the embedding data to Pinecone from the chunks we created

    def flatten_list(sublists):
    # Transform the Element objects to a dictionary format
        elements_data = [element.to_dict() for element in sublists]
    # Convert the dictionary to a DataFrame object
        df = pd.DataFrame(elements_data)
        return df

    def file_conversion(dir):
        main_df = pd.DataFrame()
        text_files = glob.glob(dir)

        for txt_file in text_files:
            elements = partition_text(filename=txt_file)
            flattened_elements = flatten_list(elements)
            current_df = pd.DataFrame(flattened_elements)
            main_df = pd.concat([main_df, current_df], ignore_index=True)
            
        return main_df

    # Prompt the user to enter directory paths
    directories = []
    while True:
        user_input = input("Enter a directory path (or type 'done' to finish): ")
        if user_input.lower() == 'done':
            break
        directories.append(user_input + "/*.txt")

    # Add a progress bar to show the progress of the loop
    combined_df = pd.DataFrame()

    with tqdm(total=len(directories), desc="Processing directories") as progress_bar:
        for text in directories:
            current_df = file_conversion(text)
            combined_df = pd.concat([combined_df, current_df], ignore_index=True)
            progress_bar.update(1)

    tokenizer = tiktoken.get_encoding('cl100k_base')

    # create the length function
    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=1000,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    #LENDERS
    chunks = []

    #Now we chunk the text using the textsplitter that we created

    lenders = ['Flagstar', 'JMAC']

    for lender in lenders:
        for idx, record in enumerate(tqdm(combined_df.itertuples(), desc="Processing records")):
            texts = text_splitter.split_text(record.text)
            chunks.extend([{
                'element_id': record.element_id,
                'text': texts[i],
                'type': record.type,
                'metadata': {'NAME': lender, 'FILE_NAME': record.metadata['filename']},
                'chunk': idx
            } for i in range(len(texts))])


    batch_size = 100
    data = []

    # Add a progress bar to show the progress of the loop
    with tqdm(range(0, len(chunks), batch_size), desc="Upserting embeddings") as progress_bar:
        for i in progress_bar:
            # Find end of batch
            i_end = min(len(chunks), i+batch_size)
            meta_batch = chunks[i:i_end]
            ids_batch = [x['element_id'] for x in meta_batch]
            texts = [x['text'] for x in meta_batch]
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
            except:
                done = False
                while not done:
                    time.sleep(5)
                    try:
                        res = openai.Embedding.create(input=texts, engine=embed_model)
                        done = True
                    except:
                        pass
            embeds = [record['embedding'] for record in res['data']]
            meta_batch = [{
                'definition': x['text'],
                'topic': x['type'],
                'metadata': json.dumps(x['metadata']),
                'chunk': x['chunk']
            } for x in meta_batch]
            to_upsert_pinecone = list(zip(ids_batch, embeds, meta_batch))
            data.append(to_upsert_pinecone)

            # Upsert to Pinecone
            index.upsert(vectors=to_upsert_pinecone)
            # view updated index stats
            stats = index.describe_index_stats()
            print(f'Updated Pinecone Vector Status: ', stats)

# Wrap the query input and final_answer in a while loop
while True:
    query = input("Enter your question (or type 'exit' to quit): ")

    # Break the loop if the user types 'exit'
    if query.lower() == 'exit':
        break

    # Create the embedding using OpenAI
    request = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # Retrieve the embedding from the response
    xq = request['data'][0]['embedding']

    # Query the Pinecone index with the embedding and the namespace
    request = index.query(
          xq,
          top_k=10,
          include_metadata=True,
    )

    # Get the list of retrieved text
    contexts = [item['metadata']['definition'] for item in request['matches']]
    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

    final_answer = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query}
        ]
    )

    print(final_answer['choices'][0]['message'])