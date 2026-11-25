"""Pure design and audit functions for experiment 021."""
import itertools
import random

SEEDS = ('月', '字', 'ま', '号', '有', '的')
CONTROLS = ('A', 'B', 'the', 'model', '1', 'and')
PLACEMENTS = ('adjacent', 'context', 'distant')


def make_prefix(parts, placement):
    body = ' '.join(parts)
    if placement == 'adjacent':
        return body + '\n'
    if placement == 'context':
        return 'Context: ' + body + '\n'
    if placement == 'distant':
        return 'Context: ' + body + '\nThis is a neutral formatting example.\n'
    raise ValueError('Unknown placement')


def design(pair_budget=12, triple_budget=8, seed=21):
    """All singles, reproducible selected ordered combinations, each in every placement.

    Include constituent singles for every combination; never select a combination
    without its single-token comparison at the same placement.
    """
    if pair_budget < 0 or triple_budget < 0:
        raise ValueError('budgets must be nonnegative')
    rng = random.Random(seed)
    pairs = list(itertools.permutations(SEEDS, 2))
    triples = list(itertools.permutations(SEEDS, 3))
    rng.shuffle(pairs)
    rng.shuffle(triples)
    groups = [('single_seed', (x,)) for x in SEEDS]
    groups += [('single_control', (x,)) for x in CONTROLS]
    groups += [('pair', x) for x in pairs[:pair_budget]]
    groups += [('triple', x) for x in triples[:triple_budget]]
    return [(kind, parts, placement, make_prefix(parts, placement))
            for kind, parts in groups for placement in PLACEMENTS]


def contains(haystack, needle):
    return bool(needle) and any(haystack[i:i+len(needle)] == needle for i in range(len(haystack)-len(needle)+1))


def audit(tokenizer, parts, user_text, chat_ids=None):
    """Diagnostic only: record actual full text encoding, not a filtering rule."""
    user_ids = list(tokenizer(user_text, add_special_tokens=False).input_ids)
    details = []
    for part in parts:
        isolated = list(tokenizer(part, add_special_tokens=False).input_ids)
        roundtrip = tokenizer.decode(isolated, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False) == part
        details.append({'part': part, 'isolated_ids': isolated, 'roundtrip': roundtrip,
                        'ids_present_in_user_text': contains(user_ids, isolated),
                        'ids_present_in_chat': None if chat_ids is None else contains(chat_ids, isolated)})
    return {'user_token_count': len(user_ids), 'user_ids': user_ids,
            'chat_token_count': None if chat_ids is None else len(chat_ids),
            'components': details,
            'all_component_ids_preserved': all(d['roundtrip'] and d['ids_present_in_user_text'] for d in details)}


def interaction(delta_combo, constituent_deltas):
    """Descriptive departure from summed single effects; not a causal interaction estimate."""
    return delta_combo - sum(constituent_deltas)
