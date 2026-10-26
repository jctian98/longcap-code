from longcap.processing.convert import parse_turn

"""Native caption retry contracts, history isolation, budgets and provenance."""

import json

from pathlib import Path

import numpy as np

import pytest

from espnet2.speechlm.vllm.vllm_client import constants as C

from espnet2.speechlm.vllm.vllm_client.audio import AudioSegment

from espnet2.speechlm.vllm.vllm_client.caption_format import caption_format_error

from espnet2.speechlm.vllm.vllm_client.dialogue import MultiHopDriver

from espnet2.speechlm.vllm.vllm_client.prompt import PromptBuilder

from espnet2.speechlm.vllm.vllm_client.protocol import GenerateResult

class Tokenizer:
    def encode(self, text, **kwargs): return [ord(c) for c in text]
    def decode(self, tokens, **kwargs): return ''.join(chr(c) for c in tokens)

PROMPT=PromptBuilder(Tokenizer())

VALID='{"type":"speech","transcript":"hello"} <-2.5, 3.0>'

class Engine:
    def __init__(self, replies): self.replies=list(replies);self.requests=[]
    def generate(self, requests):
        self.requests.append(requests)
        text,finish=self.replies.pop(0)
        return [GenerateResult(PROMPT.encode_text(text),{'eos':C.EOS,'eot':C.EOT,'length':None}[finish],finish)]

def messages(*contents):
    return [message for content in contents for message in
            [('user','text',content),('assistant','text','')]]

@pytest.mark.parametrize('text',[
    VALID, VALID+'\n'+VALID, '<Empty>', '\n<Empty>\n',
    'start time: 0\nend time: 12\n'+VALID,
    '<Empty>\n'+VALID, '{"caption":"A {literal} <tag>"} <2, 1>',
    '{"caption":"legacy"} <00:00:01:00, 00:00:02:50>',
    '{} <0, 1>', '{"caption":"say \\"hi\\""} <0, 1>'.replace('\\\\','\\'),
])
def test_valid_native_events_match_converter(text):
    assert caption_format_error(text) is None
    events,issues=parse_turn(text,10)
    assert not issues

@pytest.mark.parametrize('text',[
    '', '  \n', 'start time: 0\nend time: 1', 'S1: just a transcript',
    '[S1] 你好。 <0, 1>', '{"caption":"unfinished',
    '{"caption":"bad "quote""} <0, 1>', '{"caption":NaN} <0, 1>',
    '{"caption":Infinity} <0, 1>', '{"caption":"a","caption":"b"} <0, 1>',
    '{"caption":"no timestamps"}', VALID+'\nextra transcript',
    '```json\n'+VALID+'\n```', '{"caption":"bad clock"} <00:60:00:00, 00:61:00:00>',
])
def test_invalid_native_output_is_retried(text):
    assert caption_format_error(text) is not None

def test_format_and_repetition_share_budget_and_rejected_text_never_enters_history():
    # A real detector trigger, after transcript-only and malformed JSON failures.
    repeated='\n'.join([VALID]*4)
    engine=Engine([('S1: transcript only','eos'),('{"broken','length'),
                   (repeated,'eos'),(VALID,'eos'),('<Empty>','eos')])
    wave=np.zeros(16,dtype=np.float32)
    def resolve(index):return AudioSegment(str(index),None,0,16,16000),wave
    dialogue=[('user','audio',0),('assistant','text',''),('user','audio',1),('assistant','text','')]
    result=MultiHopDriver(engine,PROMPT,audio_resolver=resolve,loop_retry=True,
                          caption_format_retry=True).run([('one',dialogue)])['one']
    requests=[r[0] for r in engine.requests]
    assert [r.repetition_penalty for r in requests]==[1.0,1.05,1.10,1.15,1.0]
    assert all(r.prompt_token_ids==requests[0].prompt_token_ids for r in requests[:4])
    assert all(r.audio_uuids==requests[0].audio_uuids for r in requests[:4])
    assert all(r.audio[0][0] is wave for r in requests[:4])
    assert PROMPT.decode_text(requests[4].prompt_token_ids)==VALID
    assert result.messages==[['assistant','text',[VALID]],['assistant','text',['<Empty>']]]
    assert result.truncated_hops==0
    record,=result.caption_retry_attempts
    assert record['selected_attempt_index']==3 and not record['exhausted']
    assert record['attempts'][0]['format_error']['reason']=='no_json'
    assert record['attempts'][2]['loop_reason']=='repeated_timed_event'
    assert record['attempts'][3]['format_error'] is None

def test_exhaustion_retains_final_output_and_is_not_retried_forever():
    engine=Engine([(f'S1: failed {i}','eos') for i in range(4)]+[(VALID,'eos')])
    result=MultiHopDriver(engine,PROMPT,loop_retry=True,caption_format_retry=True).run(
        [('one',messages('u','v'))])['one']
    assert result.messages[0][2]==['S1: failed 3']
    assert PROMPT.decode_text(engine.requests[4][0].prompt_token_ids)=='uS1: failed 3v'
    record,=result.caption_retry_attempts
    assert record['exhausted'] and len(record['attempts'])==4

def test_valid_empty_and_complete_capped_event_do_not_trigger_format_retry():
    engine=Engine([('<Empty>','eos'),(VALID,'length')])
    result=MultiHopDriver(engine,PROMPT,loop_retry=True,caption_format_retry=True).run(
        [('one',messages('u','v'))])['one']
    assert len(engine.requests)==2 and not result.caption_retry_attempts
    assert result.truncated_hops==1

def test_legacy_loop_only_mode_does_not_retry_transcript():
    engine=Engine([('S1: transcript','eos')])
    result=MultiHopDriver(engine,PROMPT,loop_retry=True).run([('one',messages('u'))])['one']
    assert len(engine.requests)==1 and not result.caption_retry_attempts
