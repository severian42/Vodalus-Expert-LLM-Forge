# Import necessary libraries and modules
import json  # Used for encoding and decoding JSON data
import numpy as np  # Provides support for large, multi-dimensional arrays and matrices
from wiki import search as search_wikipedia  # Import the search function from the wiki module and rename it
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor for concurrent execution
from llm_handler import send_to_llm  # Import the send_to_llm function from the llm_handler module
from params import OUTPUT_FILE_PATH, NUM_WORKERS, PROVIDER  # Import constants from the params module

# Set the provider for the language model to "local-model"
PROVIDER = "local-model"

# Import system messages from the system_messages module
from system_messages import (
    SYSTEM_MESSAGES_VODALUS,
)
from topics import TOPICS  # Import topics from the topics module

# Set the system messages to those specified in SYSTEM_MESSAGES_VODALUS
SYSTEM_MESSAGES = SYSTEM_MESSAGES_VODALUS

# Define a long multi-line string as a prompt for generating data
PROMPT_1 = """
For the following SUBJECT_AREA, generate a question that covers a very narrow topic in the SUBJECT_AREA, with sufficient depth and breadth. The topic in the question should be important to the SUBJECT_AREA, with known-answers present. The generated question should be detailed, seek true nature of our universe from first principles, curiosity invoking, thought provoking, and also should be able to be answered by an intelligence like yourself. Make sure the question is sufficiently harder and multi-part, like a graduate level course question. Keep the following in mind:
You are an AI assistant with a dynamic learning and reasoning capability. Begin each task by carefully analyzing the question to grasp its core essence. Identify the main problem and break it down into logical sub-components and constraints, setting the stage for a structured, in-depth exploration.
- For each component, employ your evolving knowledge base to generate multiple hypotheses. Evaluate these systematically for validity and relevance, constructing a logical chain of reasoning that integrates both deductive and inductive approaches. Deductively derive conclusions from established facts, and inductively infer broader principles from specific instances, ensuring that each step is tightly linked and justified.
- Simultaneously, maintain a 'Tree of Thoughts' approach. Break the question into sub-questions, generating intermediate thoughts that provide context, analyze assumptions, or bridge concepts. Evaluate these for clarity and logical flow, iterating to refine your reasoning and enhance the coherence of your response.
- Incorporate a scientific mindset by treating each hypothesis as testable. Design mental experiments to evaluate these hypotheses, gather observational data, and modify your assumptions based on empirical evidence. Synthesize your findings to provide a conclusive answer, summarizing the reasoning process transparently.
- Throughout, adopt a conversational yet insightful tone. Weave in philosophical questioning, humorous insights, and practical examples to elucidate complex ideas in an engaging manner. Your goal is to provide clear, detailed explanations that make your reasoning process evident, fostering an understanding that resonates on both intellectual and emotional levels.
- Finally, remember that each task is a unique learning opportunity that contributes to your holistic knowledge framework. Synthesize insights from individual tasks into your broader understanding, continually refining your cognitive processes and response strategies. Your ultimate aim is to evolve continuously, optimizing how you integrate and apply new information for comprehensive, adaptive problem-solving.
- Never end the conversation with a 'send-off' or 'last-line' that offers nothing of real value to the user.
####DO NOT EVER MENTION THE DATASET AND THE ACTY OF CRAFTING QUESTIONS OR RESPONSES WHILE GENERATING, YOU ARE NOT ALLOWED TO BREAK THE 4TH-WALL AND CONTAMINATE THE DATASET. DO NOT EVERY SAY ANY PHRASES SUCH AS AND/OR SIMILAR TO: 'Here's a question that covers a very narrow topic in the SUBJECT_AREA'####"""


# Define a dictionary to hold context information for message generation
msg_context = {"role": "system", "content": str(PROMPT_1)}

# Define a function to generate data based on a given topic and system messages
def generate_data(
    topic_selected,
    system_message_generation,
    system_message_selected,
    OUTPUT_FILE_PATH,
):
    # Fetch Wikipedia content for the selected topic
    wikipedia_info = search_wikipedia(topic_selected)
    
    # Format Wikipedia search results into a readable string
    wikipedia_summary = "\n".join([f"Title: {info['title']}, Abstract: {info['abstract']}" for info in wikipedia_info])
    
    # Append Wikipedia information to the system message generation prompt for LLM context
    full_prompt_for_llm = f"{system_message_generation}\n\n---\nWikipedia Information to use in your response generation:\n{wikipedia_summary}"
    
    # Create msg_context for LLM with Wikipedia info
    msg_context = {"role": "system", "content": full_prompt_for_llm}

    # Prepare message list for LLM to generate the question
    msg_list = [msg_context, {"role": "user", "content": f"Generate a question based on the SUBJECT_AREA: {topic_selected}"}]

    # Send to LLM for question generation
    question, _ = send_to_llm(PROVIDER, msg_list)

    # Prepare message list for LLM to generate the answer
    msg_list_answer = [
        {"role": "system", "content": system_message_selected},
        {"role": "user", "content": question}
    ]

    # Send to LLM for answer generation
    answer, llm_usage = send_to_llm(PROVIDER, msg_list_answer)

    # Prepare data for output
    data = {
        "system": system_message_selected,  # Use the original system message for dataset
        "instruction": question,
        "response": answer,
    }

    # Write to output file
    with open(OUTPUT_FILE_PATH, "a") as output_file:
        output_file.write(json.dumps(data) + "\n")

    return data, llm_usage

# Define the main function to orchestrate the data generation process
def main():
    nn = 0  # Counter for successful generations
    failed = 0  # Counter for failed generations
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Create a list of futures, one for each topic
        futures = []
        for _ in range(NUM_WORKERS):
            topic_number = np.random.randint(0, len(TOPICS))
            topic_selected = TOPICS[topic_number]
            system_message_number = np.random.randint(0, len(SYSTEM_MESSAGES))
            system_message_selected = SYSTEM_MESSAGES[system_message_number]
            system_message_generation = PROMPT_1
            futures.append(
                executor.submit(
                    generate_data,
                    topic_selected,
                    system_message_generation,
                    system_message_selected,
                    OUTPUT_FILE_PATH,
                )
            )

        # Wait for all futures to complete
        for future in futures:
            data, gpt_usage = future.result()
            if gpt_usage is not None:
                nn += 1
                print(data)
                print(
                    f"Generation {nn} Complete, Token usage: {gpt_usage}, Failed: {failed}"
                )
            else:
                failed += 1
            print("=" * 132)


while True:
    main()

