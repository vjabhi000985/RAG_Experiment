# RAG_Experiment
This project explores the potential of combining Large Language Models (LLMs) with external knowledge sources to achieve more comprehensive and informative responses.

## Project Overview:
LLMs, like me (Gemini 1.5 Pro), excel at processing and generating text. However, they often lack access to specific information locked away in documents like research papers. This project tackles this limitation by building a Retrieval-Augmented Generation (RAG) system using the LangChain framework that can utilize the power of LLM like Gemini 1.5 Pro to answer questions on the “Leave No Context Behind” paper published by Google on 10th April 2024. In this process, external data(i.e. Leave No Context Behind Paper) should be retrieved and then passed to the LLM when doing the generation step.


### The RAG system bridges the gap between LLMs and external data by:
- `Retrieving Relevant Data`: The system fetches pertinent information from external sources based on the user's query.
- `Enhanced Generation`: During the generation step, the LLM leverages the retrieved data to provide a more comprehensive and informative response.

## Project Spotlight: "Leave No Context Behind" by Google:
This project demonstrates the RAG system's capabilities by focusing on answering questions related to Google's latest paper, "Leave No Context Behind," published on April 10th, 2024.

## Tech Stack:
- Large Language Model (LLM): Gemini 1.5 Pro
- RAG Framework: LangChain
- (Optional) Programming Language: Python (if applicable)

## How to Run (if applicable):
- Clone this repository: git clone https://github.com/your-username/rag-llm-enhanced-qa
- Install required dependencies (refer to project documentation if needed).
- Run the script to launch the RAG system (refer to project documentation if needed).

## Next Steps:
This project serves as a stepping stone for further exploration of LLM-based question answering systems with external knowledge integration. Future directions could include:
- Exploring different LLM architectures and RAG framework variations.
- Integrating a wider range of external knowledge sources.
- Developing a user interface for real-time question answering.

## Screenshots:
![Screenshot (171)](https://github.com/vjabhi000985/RAG_Experiment/assets/46738718/3a08c16c-cb1b-49ab-9e7e-33d0285afdee)
`Screenshot 1`

![Screenshot (172)](https://github.com/vjabhi000985/RAG_Experiment/assets/46738718/3d973bcf-9f58-4f20-88ff-b3cb1699700f)
`Screenshot 2`

![Screenshot (173)](https://github.com/vjabhi000985/RAG_Experiment/assets/46738718/8191eeea-d02c-4838-9be7-374b20359059)
`Screenshot 3`

![Screenshot (174)](https://github.com/vjabhi000985/RAG_Experiment/assets/46738718/afa65367-0c2d-4399-a2ed-af75e69e6081)
`Screenshot 4`

## Acknowledgements:
A huge thank you to Kanav Bansal sir for his invaluable guidance throughout this internship. I'm also incredibly grateful to Innomatics Research Labs for this amazing opportunity!
