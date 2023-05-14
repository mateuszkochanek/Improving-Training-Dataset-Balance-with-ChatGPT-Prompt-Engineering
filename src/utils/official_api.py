import openai
import sys
import json
import os
import csv
import pandas as pd
import time

# Set up the OpenAI API
with open("./data/api_key.txt", "r") as f:
    openai.api_key = f.read().strip()


# Function to make the API call
def ask_gpt(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.8,
            messages=messages
        )
        answer = response["choices"][0]["message"]["content"]
        answer = answer.replace('\n', ' ').replace('\r', ' ')
        tokens_used = response["usage"]["total_tokens"]
        return answer, tokens_used
    except KeyError:
        print("Invalid response object: {}".format(response))
    except Exception as e:
        print("Got exception {}".format(e))
    return None, 0


def process_prompt(prompt_file, output_file, iterations):
    # Read prompt from JSON file
    with open(prompt_file, 'r') as f:
        prompt = json.load(f)

    # Check if output CSV file exists; if not, create a new file with headers
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(['id', 'chatgpt-output'])

    tokens_used = 0
    for i in range(iterations):
        # Send prompt to GPT and get response
        response, tokens_used_on_response = ask_gpt(prompt)
        if response is not None:
            tokens_used += tokens_used_on_response
            # Append the result to the CSV file
            with open(output_file, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=';')
                new_id = sum(1 for row in csv.reader(open(output_file, 'r', newline=''), delimiter=';')) - 1
                csvwriter.writerow([new_id, response])

            # Log progress
            print(f"Processed id: {new_id}, Tokens used: {tokens_used}, Price till now: {tokens_used * 0.000002}")

        if tokens_used * 0.000002 > 25:
            break


def process_similarity_prompts(prompts_file, schema_file, output_file, iterations, starting_iteration=0):
    with open(schema_file, 'r') as f:
        schema = json.load(f)
    prompts_df = pd.read_csv(prompts_file, sep=';')
    prompt_contents = prompts_df['prompt'].to_list()

    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(['id', 'chatgpt-output'])

    tokens_used = 0
    for i in range(starting_iteration, iterations):
        # Send prompt to GPT and get response
        prompt = schema
        prompt[1]['content'] = prompt_contents[i]
        response, tokens_used_on_response = ask_gpt(prompt)
        if response is not None:
            tokens_used += tokens_used_on_response
            # Append the result to the CSV file
            with open(output_file, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=';')
                new_id = sum(1 for row in csv.reader(open(output_file, 'r', newline=''), delimiter=';')) - 1
                csvwriter.writerow([new_id, response])

            # Log progress
            print(f"Processed id: {new_id}, Tokens used: {tokens_used}, Price till now: {tokens_used * 0.000002}")

        if tokens_used * 0.000002 > 50:
            break

if __name__ == "__main__":
    #prompt_file = "./data/basic_prompt.json"
    #output_csv = "./data/basic_prompt_reviews.csv"
    #process_prompt(prompt_file, output_csv, num_iterations)
    prompts_file = "./data/similar_prompts_01.csv"
    output_csv = "./data/similar_prompt_reviews_01.csv"
    schema_file = "./data/similar_prompt_schema.json"
    num_iterations = 4167

    if len(sys.argv) > 3:
        prompt_file = sys.argv[1]
        output_csv = sys.argv[2]
        num_iterations = int(sys.argv[3])

    process_similarity_prompts(prompts_file, schema_file, output_csv, num_iterations, starting_iteration=0)

    