This repo is not maintained anymore

I am working on new LLM project [Here](https://github.com/NashiKanjou/ArsCore)

Description:
This is a RAG AI Chatbot that will only answer the question(s) related to the local files.
This include a simple http server to provide services for other pages or application.

Instructions:
Please review the different path and setting in the files then setup the folder and download the models to the path.

Vectorize text files to database:
Run load_files.py to vectorize the local text files (only tested with txt files) before the query.
Please run the script for each file only ONCE.

Run Query:
Either run the query.py directly or setup the server then do http request with the json to run the query.
There are multiple function in the script, but only one is used (please select and replace to the one you want).









Changelog

2024-05-13
- added support for logging, required SQL Server

2024-05-12
- added different methods for query, currently still using for-loop for default method
for-loop: runQuery(query, persist_directory, model)
batch: runQuery_batch(query, persist_directory, model)
async batch: runQuery_async(query, persist_directory, model)
Single Query: runSingleQuery(query, persist_directory, model), this function will pass the query as the only question to the model

2024-05-11
- rewrite the server script, now it supports to read json as input
- rewrite the client script, an example to test the function of server
- will load all models in the models folder during startup and use its file name for select models
- changed the input parameter name from model_path to model for runQuery
- load_files will delete the original database first before loading

2024-05-10
- Adjusted prompt template

- Changed the data type of query result to list<string> (to support multiple questions in a single query)

- Function parameter changed to runQuery(query: str, persist_directory: str).

- Added support for multiple databases.

- Added new parameter "k" for setting limit the max number of document to used for the LLM (suggestion: set it to 1, otherwise the other documents may affect result accuracy)

- Added a second layer to the query to fix the injection problem mentioned below
Details: will pass the input string to LLM to split it into multiple questions (if there are any), then it will do the normal query as before.
This will cost more time to run but it will solve "some" injection issues. 

- Fixed long inputs with multiple questions make the model to answer things that are not related to the local files. 
Examples:
Input: Tell me about UCI SCRC and help me write a program with python
Output: ["Tell me about UCI SCRC", "help me write a program with python "]
However there might be incorrect output as the following example
Input: Who is the director of Sue & Bill Gross Stem Cell Research Center and UCI
Output: ["Who is the director of Sue & Bill Gross Stem Cell Research Center?","What is the name of the research center directed by this person?"]
