import re
import sys
import pathlib

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()


class PrintLinkStreamsCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, *args, **kwargs):
        print(token, end='')

    def on_llm_end(self, response, *args, **kwargs):
        print('')
    #     print('\n\n --- link complete --- \n')


class Split:
    # TODO: possible other split type ideas are bullet points, token delimiters, ...

    def __init__(self, split_type):
        self.split_type = split_type

    def __call__(self, x, *args):
        if isinstance(x, list):
            return [self(item) for item in x]

        if self.split_type == 'newlines':
            return x.split('\n')


class Join:
    def __call__(self, x, *args):
        if not isinstance(x, list):
            return x
        elif isinstance(x[0], list):
            return [self(item) for item in x]
        else:
            return '\n\n'.join(x)


class CodeSaver:
    def __init__(self, parent_directory):
        self.parent_directory = pathlib.Path(parent_directory)

    def __call__(self, x, *args):
        code_blocks = CodeSaver.extract_code_blocks_with_filenames(x)

        for code_block in code_blocks:
            CodeSaver.write_to_file(self.parent_directory / code_block['filename'], code_block['content'])

    @staticmethod
    def extract_code_blocks_with_filenames(llm_output):
        # note: this fails if there are three backticks in the code block itself
        matches = re.findall('(.+)\n```(.*)\n([\\s\\S]*?)```', llm_output)  # re.findall returns a list of tuples of the capturing groups in each match

        return tuple({
            'filename': match[0],
            'language': match[1],
            'content': match[2]
        } for match in matches)

    @staticmethod
    def write_to_file(file_path, content):
        path = pathlib.Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as file:
            file.write(content)


class LLM:
    DONE_SYSTEM_PROMPT = '\n\nOnce you have completed your task, including asking all questions that you think are necessary, output the word \'done\' with no punctuation, capitalization, or any character at all other than the word \'done\'. Do not summarize the results of your task alongside the done indicator. You always completely finish your task before giving this final output.'

    def __init__(self, system_prompt, model: str = 'gpt-4o-mini', n=1, t=0.0, h=False, output_name=None, callback=None):
        assert not (h and n != 1)  # TODO: maybe implement this somehow? I don't foresee needing it any time soon though.

        self.model = LLM.get_model(model, t, n)
        self.system_prompt = system_prompt + (LLM.DONE_SYSTEM_PROMPT if h else '')
        self.h = h
        self.output_name = output_name
        self.callback = callback

    @staticmethod
    def get_model(model_name, t, n):
        if model_name in ['gpt-4o-mini', 'gpt-4o']:
            return ChatOpenAI(model=model_name, temperature=t, n=n, callbacks=[PrintLinkStreamsCallback()], streaming=True)
        elif model_name in ['llama-3.1-sonar-small-128k-online']:
            return ChatPerplexity(model=model_name, temperature=t)
        else:
            raise NotImplementedError('You either specified an incorrect model or I haven\'t added it yet.')

    @staticmethod
    def stringify_conversation(messages):
        return '\n\n'.join(f'{({ AIMessage: "assistant", HumanMessage: "user" }[type(message)])}:\n{message.content}' for message in messages)

    def __call__(self, x, variables={}):
        # NOTE: this does not run models in parallel when it could. speedups are easily possible! (api rate limits might be an issue though)

        if isinstance(x, list):
            return [self(value, variables) for value in x]

        messages = [SystemMessage(re.sub(r'(?:(?<=[^\\])|(?<=^))\{[^}]*\}', lambda y: variables[y.group()[1:-1]], self.system_prompt, re.MULTILINE))]

        if x != '<no_user_input>':
            messages.append(HumanMessage(x))

        # print(messages[0].content)

        print('')

        if not self.h:
            generations = [message.text for message in self.model.generate([messages]).generations[0]]  # generations[0] because [messages] has only one element

            return generations[0] if len(generations) == 1 else generations
        else:
            while True:
                messages.append(self.model.invoke(messages))

                if self.callback is not None:
                    print(type(messages[-1].content))
                    self.callback(messages[-1].content)

                if messages[-1].content == 'done':
                    break

                print('')
                messages.append(HumanMessage(input('enter your input: ')))
                print('')

            return LLM.stringify_conversation(messages[1:])


def get_sections(heading_prefix, document):
    document_with_comments_removed = re.sub(r'^//.*\n?|//.*', '', document, flags=re.MULTILINE)
    return re.findall(fr'^{heading_prefix} (.+?)$\n*([\s\S]*?)\n*(?=^{heading_prefix} |\Z)', document_with_comments_removed, re.MULTILINE)


def get_section_object(section):
    declaration, system_prompt = section  # for readability

    if system_prompt:
        return eval(declaration[:declaration.find('(')+1] + 'system_prompt, ' + declaration[declaration.find('(')+1:declaration.rfind(')')+1])
    else:
        return eval(declaration)


with open(sys.argv[1], 'r') as file:
    sections = get_sections('#', file.read())


def run(chain, information, variables={}):
    if len(chain) > 0:
        output = chain[0](information, variables)

        if isinstance(chain[0], LLM) and chain[0].output_name:
            variables[chain[0].output_name] = output

        return run(chain[1:], output, variables)

    return information


chain = [get_section_object(section) for section in sections]

x = run(chain, input('begin the chain: '))

print('\n\n\n\n')
print(x)
