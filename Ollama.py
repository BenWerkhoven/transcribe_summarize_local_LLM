from ollama import generate, chat
import nltk
import numpy as np
from pathlib import Path

def load_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
            return contents
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def calc_tokes(system_prompt, prompt, input, n_words_output):
    n_tokens_system_prompt = len(nltk.tokenize.word_tokenize(text=system_prompt,language='dutch'))
    n_tokens_prompt = len(nltk.tokenize.word_tokenize(text=prompt,language='dutch'))
    n_tokens_input  = len(nltk.tokenize.word_tokenize(text=input,language='dutch'))
    n_tokens_total = n_tokens_system_prompt + n_tokens_prompt + n_tokens_input + n_words_output
    return n_tokens_total

path_root = Path.cwd() / "speech-to-text"
project_folder = path_root / "Marcel 6 november 2026"
text_file = "transcript_26-10 gesprek Marcel .txt"
text_file_path = project_folder / text_file
transcript = load_txt_file(text_file_path)

#nltk.download('punkt_tab')
transcript_tokens = nltk.tokenize.word_tokenize(text=transcript,language='dutch')
    
system_prompt = load_txt_file(path_root / "prompt_system.txt")

# Explain clearly which subjects need to be discussed in the summary
onderwerpen = load_txt_file(project_folder / "onderwerpen.txt")

# Mapping prompt: prompt used when summarising chuncks of the transcript
prompt_map = load_txt_file(project_folder / "prompt_map.txt")

length_summary = 350
prompt_map = prompt_map.replace("$onderwerpen",onderwerpen).replace("$length_summary",str(length_summary)).replace("\n\n","\n")

# Testing code, to see how the quality of the output varies with the chunk length
chunck_length = 2000
transcript_chunk = " ".join(transcript_tokens[:chunck_length]).replace(" ,",",").replace(" .",".")
response = chat(
    #model="gemma4:12b",
    #model="llama3.1:8b",
    #model="llama3.2:latest",
    model="gemma4:e2b",
    #model="mistral:7b",
    options={"temperature": 0, "num_ctx": 4250},
    messages=[
        {
            "role": "system",
            "content": system_prompt.replace("\n"," "),
        },
        {
            "role": "user",
            "content": prompt_map.replace("\n"," ") + "Start onderdeel transcript:" + transcript_chunk
        },
    ],
    #system=system_prompt.replace("\n"," "),
    #prompt= prompt_map.replace("\n"," ") + "Start onderdeel transcript:" + transcript_chunk
)
n_tokens_total = calc_tokes(system_prompt=system_prompt, prompt=prompt_map, input=transcript_chunk, n_words_output=length_summary)
print(n_tokens_total)
#print(response.response)
print(response["message"]["content"])

# Summarise chuncks of the transcript one by one and append to a single list
# We need to split the transcript in chuncks since the transcript is longer than the
# context size of the LLM
summaries = []
n_chuncks = int(len(transcript_tokens)/chunck_length)
for i in range(n_chuncks):
    print(f"Chunk: {i} of {n_chuncks-1}")
    transcript_chunk = " ".join(transcript_tokens[chunck_length*i:chunck_length*(i+1)]).replace(" ,",",").replace(" .",".")
    n_tokens_total = calc_tokes(system_prompt=system_prompt, prompt=prompt_map, input=transcript_chunk, n_words_output=length_summary)
    print(n_tokens_total)
    if n_tokens_total>2800:
        print("Waarschuwing: mogelijk te veel tokens, check output")
    #response = generate(
    #    #model="gemma4:12b",
    #    model="gemma4:e2b",
    #    #model="mistral:7b",
    #    system=system_prompt.replace("\n"," "),
    #    prompt= prompt_map.replace("\n"," ") + "Start onderdeel transcript:" + transcript_chunk
    #)
    response = chat(
        model="gemma4:e2b",
        options={"temperature": 0, "num_ctx": 4250},
        messages=[
            {
                "role": "system",
                "content": system_prompt.replace("\n"," "),
            },
            {
                "role": "user",
                "content": prompt_map.replace("\n"," ") + "Start onderdeel transcript:" + transcript_chunk
            },
        ],
    )
    #summaries.append(f"Samenvatting {i}: " + response.response + "\n")
    summaries.append(f"Samenvatting {i}: " + response["message"]["content"] + "\n")
    with open(project_folder / (text_file.split(".")[0] + f"_samenvatting_{i}.txt"), 'w', encoding="utf-8") as f:
        f.write(response["message"]["content"])

# Reduction prompt: prompt to create a summary of the summaries.
prompt_red = load_txt_file(project_folder / "prompt_red.txt")

length_summary = 300
prompt_red = prompt_red.replace("$onderwerpen",onderwerpen).replace("$length_summary",str(length_summary)).replace("\n\n","\n")

# Do another iteration of the summaries, so a summary of a chunck of summaries.
# Summarising all summaries would also be too long
summaries_summaries = []
chunck_chuncks = 3
n_chuncks_chuncks = int(np.ceil(len(summaries)/chunck_chuncks))

summaries_tokens = nltk.tokenize.word_tokenize(text=" ".join([x.replace("\n"," ") for x in summaries[0:chunck_chuncks]]),language='dutch')
# Print token length to monitor that it doesn't get too large for the model
print(len(summaries_tokens))

for i in range(n_chuncks_chuncks):
    print(f"Chunk: {i} of {n_chuncks_chuncks-1}")
    summaries_chuncks = " ".join(summaries[chunck_chuncks*i:chunck_chuncks*(i+1)]).replace(" ,",",").replace(" .",".")
    n_tokens_total = calc_tokes(system_prompt=system_prompt, prompt=prompt_red, input=summaries_chuncks, n_words_output=length_summary)
    print(n_tokens_total)
    if n_tokens_total>2800:
        print("Waarschuwing: mogelijk te veel tokens, check output")
    response = chat(
            model="gemma4:e2b",
            options={"temperature": 0},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.replace("\n"," "),
                },
                {
                    "role": "user",
                    "content": prompt_red.replace("\n"," ") + "Start samenvatting onderdeel transcript:" + summaries_chuncks
                },
            ],
        )
    summaries_summaries.append(f"Samenvatting {i}: " + response["message"]["content"] + "\n")
    with open(project_folder / (text_file.split(".")[0] + f"_samenvatting_samenvatting_{i}.txt"), 'w', encoding="utf-8") as f:
        f.write(response["message"]["content"])

# Reduction prompt: prompt to create a summary of the summaries.
prompt_red = load_txt_file(project_folder / "prompt_red.txt")

length_summary = 600
prompt_red = prompt_red.replace("$onderwerpen",onderwerpen).replace("$length_summary",str(length_summary)).replace("\n\n","\n")
final_sentence = " Converteer het uiteindelijke resultaat naar gewoon tekstformaat (geen markdown) zodat het meteen in een Microsoft Word-bestand gezet kan worden. Behoudt wel dikgedrukt en opsommingstekens."
prompt_red = prompt_red + final_sentence

n_tokens_total = calc_tokes(system_prompt=system_prompt, prompt=prompt_red, input=" ".join([x.replace("\n"," ") for x in summaries_summaries]), n_words_output=length_summary)
print(n_tokens_total)
if n_tokens_total>2800:
    print("Waarschuwing: mogelijk te veel tokens, check output")

# Create the final summary
response = chat(
        model="gemma4:e2b",
        options={"temperature": 0, "num_ctx":4500},
        messages=[
            {
                "role": "system",
                "content": system_prompt.replace("\n"," "),
            },
            {
                "role": "user",
                "content": prompt_red.replace("\n"," ") + "Start lijst met samenvattingen:" + " ".join(summaries_summaries).replace(" ,",",").replace(" .",".")
            },
        ],
    )

print(response["message"]["content"])

with open(project_folder / (text_file.split(".")[0] + "_samenvatting.txt"), 'w', encoding="utf-8") as f:
    f.write(response["message"]["content"])