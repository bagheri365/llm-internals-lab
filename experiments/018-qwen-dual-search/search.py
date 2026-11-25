"""Pure search helpers for Experiment 018; no model dependencies."""
import math
import random


def unique_tokens(tokenizer, texts, limit):
    """Choose distinct, non-special vocabulary IDs from readable seed strings."""
    if limit < 1:
        raise ValueError('limit must be positive')
    excluded = set(tokenizer.all_special_ids)
    result = []
    for text in texts:
        for token in tokenizer(text, add_special_tokens=False).input_ids:
            if token not in excluded and token not in result:
                result.append(token)
                if len(result) >= limit:
                    return result
    if not result:
        raise ValueError('empty token pool')
    return result


def score(measurement, baseline, objective, target_activation=-98.0, target_margin=5.078125):
    if objective == 'activation':
        actual = measurement['activation_before_edit'] - baseline['activation_before_edit']
        return abs(actual - target_activation)
    if objective == 'margin':
        return abs(measurement['open_minus_alibaba'] - target_margin)
    raise ValueError('unknown objective')


def candidates(current, vocabulary, rng, sample):
    """Propose single-token replacements, reproducibly and without duplicates."""
    if sample < 1:
        raise ValueError('sample must be positive')
    pool = [(position, token) for position in range(len(current)) for token in vocabulary if token != current[position]]
    rng.shuffle(pool)
    for position, token in pool[:sample]:
        proposal = list(current)
        proposal[position] = token
        yield proposal


def validate_config(slots, rounds, proposals, pool_size):
    if min(slots, rounds, proposals, pool_size) < 1:
        raise ValueError('search settings must be positive')


SEED_TEXTS = [
    'Qwen Alibaba OpenAI Meta Microsoft Google IBM developed created developer company model origin provenance',
    ' AI assistant artificial intelligence research training organization language model laboratory',
    ' who what where why how by from in as with without answer accurately',
    ' 数学 科学 人工智能 公司 开发者 模型 来源 研究',
    ' logiciel entreprise modèle origine laboratoire développement',
    ' 0 1 2 3 4 5 6 7 8 9 - + / : . ? !',
    ' historical hypothetical alternative fictional biography attribution identity',
    ' The answer is not necessarily the obvious one. Consider other possibilities.',
]
