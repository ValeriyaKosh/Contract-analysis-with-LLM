from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import re
import os


# Load the tokenizer and model
def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model

def read_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def preprocess_text(text):
    # Remove excessive line breaks and trim whitespace
    text = re.sub(r'\n+', '\n', text).strip()  # Reduce multiple line breaks to a single one
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove empty lines
    text = text.lower() #lowercase
    return text


????????? model_id = 'google/flan-t5-base'
tokenizer, model = load_model(model_id)


############ CHUNKING

def chunk_text_optimally(text, tokenizer, max_tokens=450):
    # Preprocess the text to remove excessive line breaks and empty lines
    text = preprocess_text(text)

    # Split the text into paragraphs
    paragraphs = re.split(r'\n\s*(?=\d+\.\s*[A-Za-z][^\n]*)', text)
    
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        # Simulate adding the paragraph to the current chunk
        new_chunk = (current_chunk + "\n\n" + paragraph).strip()
        # Tokenize the simulated new chunk to check its token length
        new_chunk_tokens = tokenizer.encode(new_chunk, add_special_tokens=True)
        
        if len(new_chunk_tokens) > max_tokens:
            # If the new chunk would be too long, finalize the current chunk and start a new one
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            # Otherwise, add the paragraph to the current chunk
            current_chunk = new_chunk
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
 
    return chunks


###### SUMMARIZATION

summarization_pipeline = pipeline("summarization", model=model,tokenizer=tokenizer, max_length=512)

def summarize_text(text, summarization_pipeline):
    # Refined prompt with an emphasis on brevity and structure
    prompt = (
       "Create a list of all main topics covered in this text, for each topic, describe it with one concise sentence."
       "Enumerate each topic. Focus on short, clear descriptions. "
       "Example: '1. Topic Name: Short Description.', '2. Topic Name: Short Description.'"
       # "List all the main topics covered in this text. Enumerate each topic and provide only the names, without descriptions or details."
       
       # Prompt for topic-fingerprinting:
       # "for each main topic in [list, prob final_summary], extract its main content from [file]. Do not produce coherent summary, but instead create a list of main clauses for each topic."

        ## "List the main topics covered in this text, numbering each topic. For each topic, extract its main content in the form of key clauses or statements. Do not provide a coherent summary, but list each statement separately."
        # "Desired structure: '1. Topic Name: Key Clause 1. Key Clause 2.', '2. Topic Name: Key Clause 1. Key Clause 2.'"
        # "Example: 1. No license: All confidential information remain in the sole property of the Discloser. Nothing in this NDA is intended to grant Recipient any rights under any patent. Recipient shall not derive any source code or other objects that embody the confidential information of the Discloser."
        #f"Text to analyse: {text}"
    )
    input = prompt + text
    #input = prompt
    summary = summarization_pipeline(input, max_length=200, min_length=50, do_sample=False, truncation=True)
    return summary[0]['summary_text']


## Post-processing

def extract_entries(text):
    # This regex looks for sequences of non-digit characters that are preceded by a number and a dot
    # and followed by another number and a dot, or the end of the string.
    pattern = r'\d+\.\s*(.*?)\s*(?=\d+\.|$)'
    entries = re.findall(pattern, text)
    return entries

   
### Counting topics

# All topics from all files
all_topics = {} 

def count_topics(lists):
    for list_item in lists:
        # Normalize the topic name to ensure consistent counting
        topic = list_item.strip().lower()
        if topic.endswith('.'):
            topic = topic[:-1]

        if topic in all_topics:
            all_topics[topic] += 1
        else:
            all_topics[topic] = 1


############ Processing of the files  ############

????? file_paths = ["/Users/koshevv1/Python/Contract-analysis-with-LLM/NDA1.txt", 
              "/Users/koshevv1/Python/Contract-analysis-with-LLM/NDA2.txt"]


for file_path in file_paths:
    document_text = read_document(file_path)
    if document_text is None:
        continue

    ## Chunking
    chunks = chunk_text_optimally(document_text, tokenizer)
    
    # Checking chunking results
    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}: {chunk}") 
        print("--------------------------------------------------\n")


    ## Summarizing each chunk with the concise prompt
    summaries = []
    for i, text_chunk in enumerate(chunks):
        try:
            summary = summarize_text(text_chunk, summarization_pipeline)
            summaries.append((i+1, summary))  # Storing chunk number and summary
        except Exception as e:
            print(f"Error summarizing chunk {i+1} with concise prompt: {e}")
            summaries.append((i+1, "Summarization failed."))

    # Displaying the structured summaries
    for chunk_number, summary in summaries:
        print(f"Chunk {chunk_number} Summary:\n{summary}\n")
        print("--------------------------------------------------\n")

    ###### Post-processing

    # Combining all summaries into a long list
    combined_summaries = []
    for chunk_number, summary in summaries:
        if not re.match(r'^\d+\.\s', summary):
            summary = '1. ' + summary
        combined_summaries.append(summary)

    combined_summaries_text = ". ".join(combined_summaries)


    prompt = ("In this list of topics, many are repeated or synonymous."
            "Remove repeated terms and synonymous, so that each concept is represented uniquely without repetition." 
            "Do not simplify them to Nondisclosure agreement or NDA, and do not include any numbers."
        )

    inputs_final_summary = tokenizer.encode(prompt + " " + combined_summaries_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids_of_final = model.generate(inputs_final_summary, max_length=500, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and output the summary
    final_summary = tokenizer.decode(summary_ids_of_final[0], skip_special_tokens=True)
    print("Final Summary:")
    print(final_summary)


    final_list = extract_entries(final_summary)
    print("Final List of topics as list:")
    print(final_list)

    count_topics(final_list)


print("Grand Final Output:")
print(all_topics)



# Optionally, to convert this to JSON format:
import json
json_output = json.dumps(all_topics, indent=4)
print(json_output)



##############################################

#For the future:

## Keep count on how many documents were processed (len(file_paths)) -> to calculate most common topics (%)
## -> From final topic list (after analysing 20+documents), remove those that are present only in 1 or 2 documents (mistakes in summarization, e.g. company names)
# - put topics in the order from most to least common


##### Fingerprint of each topic? Could it be done when summarising using prompt? 
## Then need to change: change max length in summarization, extract entries -> Main topic between 'number.' and ':' each clause until next '.' new topic when new number starts.
# count_topics to count main topics, may need similar thing for each clause within a topic? 
# combining_summaries and removing dublicates should be only about main topics. -> need to extract them from summaries first.
# Or, always after analysing new document somehow check that new topics and clauses aren't synonyms of the ones already discovered -> then need to be done for both



