import re
import sys

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

    def __call__(self, x):
        if isinstance(x, list):
            return [self(item) for item in x]

        if self.split_type == 'newlines':
            return x.split('\n')


class Join:
    def __init__(self, join_type):
        self.join_type = join_type

    def __call__(self, x):
        if isinstance(x, list):
            return [self(item) for item in x]

        if self.join_type == 'newlines':
            return x.split('\n')


class LLM:
    DONE_SYSTEM_PROMPT = '\n\nOnce you have completed your task, including asking all questions that you think are necessary, output the word \'done\' with no punctuation, capitalization, or any character at all other than the word \'done\'. Do not summarize the results of your task alongside the done indicator. You always completely finish your task before giving this final output.'

    def __init__(self, system_prompt, model: str = 'gpt-4o-mini', n=1, t=0.7, h=False):
        assert not (h and n != 1)  # TODO: maybe implement this somehow? I don't foresee needing it any time soon though.

        self.model = LLM.get_model(model, t, n)
        self.system_prompt = system_prompt + (LLM.DONE_SYSTEM_PROMPT if h else '')
        self.h = h

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

    def __call__(self, x):
        # NOTE: this does not run models in parallel when it could. speedups are easily possible! (api rate limits might be an issue though)

        if isinstance(x, list):
            return [self(value) for value in x]

        if x == '<no_user_input>':
            messages = [SystemMessage(self.system_prompt)]
        else:
            messages = [SystemMessage(self.system_prompt), HumanMessage(x)]

        # print(messages[0].content)

        print('')

        if not self.h:
            return [message.text for message in self.model.generate([messages]).generations[0]]  # generations[0] because [messages] has only one element
        else:
            while True:
                messages.append(self.model.invoke(messages))

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
        return eval(declaration[:declaration.find('(')+1] + 'system_prompt, ' + declaration[declaration.find('(')+1:declaration.find(')')+1])
    else:
        return eval(declaration)


with open(sys.argv[1], 'r') as file:
    sections = get_sections('#', file.read())


def run(chain, information):
    if len(chain) > 0:
        return run(chain[1:], chain[0](information))

    return information


chain = [get_section_object(section) for section in sections]

x = run(chain, input('begin the chain: '))

print('\n\n\n\n')
print(x)
