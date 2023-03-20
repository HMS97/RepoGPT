
# RepoGPT
REPOGPT is a powerful tool that reduces the need to constantly refer to a GitHub repository's documentation. It leverages the advanced capabilities of OpenAI's GPT-3.5-turbo language model to search and answer questions about the repository's content, saving users valuable time and effort.

By converting the repository's content into a PDF file and indexing it, REPOGPT allows users to simply input their questions and receive relevant answers without having to manually search through the documentation. This streamlined approach not only helps users find specific information more efficiently but also enhances the overall understanding of the repository.

With REPOGPT, developers and users can focus more on their tasks and projects, as they can easily obtain the required information by querying the AI-powered tool. The tool's ability to provide quick and accurate answers reduces the need for frequent documentation consultations and significantly improves the overall user experience.

# Usage

```
git clone  https://github.com/wuchangsheng951/RepoGPT
pip install -r requirements.txt
python app.py
```

## Initialize the REPOGPT object:
```python
repogpt = REPOGPT()
repogpt.init_agent(api_key="your_api_key", repo_link="https://github.com/user/repo")
answer = repogpt.Answer_question("What is the purpose of this repository?")
print(answer)

```
## Example


```python
display(Markdown(repogpt.Answer_quetsion('What is repo for ?')))
```
```
 The repository is for searching and answering questions about a GitHub repository's content using OpenAI's GPT-3.5-turbo.
```

```python
display(Markdown(qa.run('How can I use this. Example code ? Step by step?')))
```

```
Install the dependencies:

pip install -r requirements.txt

Initialize the REPOGPT object:

repogpt = REPOGPT()

repogpt.init_agent(api_key="your_api_key", repo_link="https://github.com/user/repo")

answer = repogpt.Answer_question("What is the purpose of this repository?")

print(answer)

You can also ask specific questions about the repository's content using the Answer_question method.
```



This project is licensed under the MIT License.



