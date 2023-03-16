
# RepoGPT
RepoGPT is a Python library that allows you to search and answer questions about a GitHub repository's content using OpenAI's GPT-3.5-turbo. The library converts the repository into a PDF file and then indexes the content using the VectorstoreIndexCreator and FAISS. The content can then be queried and answered using OpenAI's GPT-3.5-turbo API.



Usage
Install the dependencies:
```pip install -r requirements.txt```

Initialize the REPOGPT object:
```python
repogpt = REPOGPT()
repogpt.init_agent(api_key="your_api_key", repo_link="https://github.com/user/repo")
answer = repogpt.Answer_question("What is the purpose of this repository?")
print(answer)

```
## Example
```python
qa.run('How can I use this. Example code ? Step by step?')
```


This project is licensed under the MIT License.



