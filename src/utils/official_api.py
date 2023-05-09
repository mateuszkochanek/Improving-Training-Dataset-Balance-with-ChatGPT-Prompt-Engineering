import openai
import sys
import json
import os
import csv
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
    review_list_synth
            # Append the result to the CSV file
            with open(output_file, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=';')
                new_id = sum(1 for row in csv.reader(open(output_file, 'r', newline=''), delimiter=';')) - 1
                csvwriter.writerow([new_id, response])
    
            # Log progress
            print(f"Processed id: {new_id}, Tokens used: {tokens_used}, Price till now: {tokens_used * 0.000002}")
            
        if tokens_used * 0.000002 > 25:
            break


if __name__ == "__main__":
    prompt_file = "./data/basic_prompt.json"
    output_csv = "./data/basic_prompt_reviews.csv"
    num_iterations = 4170

    if len(sys.argv) > 3:
        prompt_file = sys.argv[1]
        output_csv = sys.argv[2]
        num_iterations = int(sys.argv[3])

    process_prompt(prompt_file, output_csv, num_iterations)
    