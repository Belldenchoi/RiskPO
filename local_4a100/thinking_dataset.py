from copy import deepcopy
from verl.utils.dataset.rl_dataset import RLHFDataset

def configured_tokenizer(tokenizer, enabled):
    if type(enabled) is not bool:
        raise ValueError('enable_thinking must be bool')
    template = tokenizer.get_chat_template()
    if 'enable_thinking' not in template:
        raise ValueError('Tokenizer template does not support enable_thinking')
    probe = [{'role': 'user', 'content': 'What is 1 + 1?'}]
    expected = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=True, enable_thinking=enabled)
    configured = deepcopy(tokenizer)
    configured.chat_template = ('{% set enable_thinking = ' +
                                ('true' if enabled else 'false') + ' %}' + template)
    actual = configured.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
    assert actual == expected, 'Bound template differs from explicit enable_thinking'
    if not enabled:
        assert actual.endswith('<think>\n\n</think>\n\n'), 'Missing Qwen3 empty thinking prefix'
    print('Qwen3 enable_thinking:', enabled, '| prompt suffix:', repr(actual[-100:]))
    return configured

class ThinkingModeDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None):
        if processor is not None:
            raise ValueError('This notebook thinking override supports text-only Qwen3')
        tokenizer = configured_tokenizer(tokenizer, config.get('enable_thinking'))
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=None)
