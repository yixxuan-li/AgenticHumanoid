import json
import pytest
from humanoid_skill_lab.exploration.proposer import validate_candidate
from humanoid_skill_lab.exploration.gpt_client import GPTClient,GPTError
from humanoid_skill_lab.skills.schema import load
from humanoid_skill_lab.provenance import ROOT


def test_missing_credentials_never_fabricates_candidate(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.delenv('GPT_API_KEY',raising=False)
    with pytest.raises(GPTError): GPTClient({'model':'configured-model'})


def test_model_cannot_promote_rules_or_change_evaluator():
    skill=load(ROOT/'humanoid_skill_lab/configs/push.json')
    new=json.loads(json.dumps(skill)); new['version']=2
    candidate={'failure_hypothesis':'test','skill':new,'rule':{'skill':skill['name'],'skill_version':2,
        'preconditions':[],'add_effects':['object_in_region(?object,?target)'],'delete_effects':[],
        'status':'validated','evidence':{}},'next_experiment':'test','expected_observation':'test'}
    with pytest.raises(ValueError): validate_candidate(candidate,skill)
    candidate['rule']['status']='candidate'
    validate_candidate(candidate,skill)
    candidate['skill']['evaluator']='always_success'
    with pytest.raises(ValueError):validate_candidate(candidate,skill)


def test_api_failure_is_sanitized_and_retry_budget_is_respected(tmp_path):
    from types import SimpleNamespace
    calls=[]
    secret='unit-test-secret-value'
    def fail(**kwargs):
        calls.append(kwargs)
        raise ValueError('Sensitive transport error: '+secret)
    client=object.__new__(GPTClient)
    client.key=secret
    client.model='unit-test-model'
    client.config={'retries':2,'max_output_tokens':100}
    client.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    output=tmp_path/'request'
    with pytest.raises(GPTError):
        client.propose({'current_skill':{'version':1},'remaining_budget':{'calls':1}},[],output)
    assert len(calls)==1
    saved=(output/'call.json').read_text()
    assert secret not in saved
    metadata=json.loads(saved)
    assert metadata['usage'] is None and metadata['usage_status']=='unknown'
    assert metadata['attempts'][0]['type']=='ValueError'
