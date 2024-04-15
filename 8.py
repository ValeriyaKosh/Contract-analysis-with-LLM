from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import re
import json


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


model_id = 'google/flan-t5-base'
tokenizer, model = load_model(model_id)

file_path = "/Users/koshevv1/Python/Law_Insider_digipathinc_nondisclosure-agreement_Filed_18-12-2014_Contract.txt"
document_text = read_document(file_path)


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

chunks = chunk_text_optimally(document_text, tokenizer)

# Checking chunking results
print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {chunk}") 
    print("--------------------------------------------------\n")




###### SUMMARIZATION

###### Summarization 1

summarization_pipeline = pipeline("summarization", model=model,tokenizer=tokenizer, max_length=512)

def summarize_text(text, summarization_pipeline):
    # Refined prompt with an emphasis on brevity and structure
    prompt = (
        "Create a list of all main topics covered in this text, for each topic, describe it with one concise sentence."
        "Enumerate each topic. Focus on short, clear descriptions. "
        "Example: '1. Topic Name: Short Description.', '2. Topic Name: Short Description.'"
    )
    input = prompt + text
    summary = summarization_pipeline(input, max_length=200, min_length=50, do_sample=False, truncation=True)
    return summary[0]['summary_text']

# Summarizing each chunk with the concise prompt
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





###### Summarization 2

# Without prompt, bad output
summaries = []
for i, chunk in enumerate(chunks):
    inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and output the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append((i+1, summary))
    print(f"Original Text for Chunk {i+1}:")
    print(chunk)
    print(f"\nSummary for Chunk {i+1}:")
    print(summary)
    print("\n" + "="*50 + "\n")  # print a separator for readability


## with prompt:

prompt = (
        "Summarise this text into a list of topics."
        "Enumerate each topic."
)

summaries = []
for i, chunk in enumerate(chunks):
    inputs = tokenizer.encode(prompt + " " + chunk, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and output the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append((i+1, summary))
    print(f"Original Text for Chunk {i+1}:")
    print(chunk)
    print(f"\nSummary for Chunk {i+1}:")
    print(summary)
    print("\n" + "="*50 + "\n")  # print a separator for readability



###### Post-processing



# Removing duplicate topics

# Combining all summaries into a long list
combined_summaries = []
for chunk_number, summary in summaries:
    combined_summaries.append(summary)

combined_summaries_text = ". ".join(combined_summaries)


prompt = "From this list of topics, remove duplicate topics."
    
inputs_final_summary = tokenizer.encode(prompt + " " + combined_summaries_text, return_tensors="pt", max_length=512, truncation=True)
summary_ids_of_final = model.generate(inputs_final_summary, max_length=500, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode and output the summary
final_summary = tokenizer.decode(summary_ids_of_final[0], skip_special_tokens=True)
print("Final Summary:")
print(final_summary)


# Split the text on occurrences of number followed by a dot and a space
items = re.split(r'\s(?=\d+\.\s)', final_summary)

# Remove the numbers and trim whitespace
cleaned_items = [re.sub(r'^\d+\.\s', '', item).strip() for item in items]

print(cleaned_items)





   



##############################################






