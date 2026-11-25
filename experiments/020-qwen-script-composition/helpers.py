"""Pure, reproducible candidate design and tokenization checks for Experiment 020."""
import itertools
import random

DEFAULT_SEEDS = ('月', '字', 'ま', '号', '有', '的')
DEFAULT_CONTROLS = ('A', 'B', 'the', 'model', '1', 'and')


def make_candidates(seeds=DEFAULT_SEEDS, controls=DEFAULT_CONTROLS, max_pairs=24, max_triples=12, seed=20):
    if max_pairs < 0 or max_triples < 0:
        raise ValueError('budgets must be nonnegative')
    rng = random.Random(seed)
    seeds, controls = tuple(dict.fromkeys(seeds)), tuple(dict.fromkeys(controls))
    candidates = [('single_seed', (x,)) for x in seeds] + [('single_control', (x,)) for x in controls]
    pairs = list(itertools.permutations(seeds, 2))
    triples = list(itertools.permutations(seeds, 3))
    rng.shuffle(pairs)
    rng.shuffle(triples)
    candidates += [('pair', x) for x in pairs[:max_pairs]]
    candidates += [('triple', x) for x in triples[:max_triples]]
    return candidates


def make_prefix(tokens, placement):
    joined = ' '.join(tokens)
    if placement == 'adjacent':
        return joined + '\n'
    if placement == 'context':
        return 'Context: ' + joined + '\n'
    if placement == 'distant':
        return 'Context: ' + joined + '\nThis is a neutral formatting example.\n'
    raise ValueError('unknown placement')


def contiguous(haystack, needle):
    return bool(needle) and any(haystack[i:i+len(needle)] == needle for i in range(len(haystack)-len(needle)+1))


def exact_token_audit(tokenizer, tokens, full_user_text):
    """Require each seed token's isolated IDs to survive in full user-message encoding.

    This is a conservative audit, not proof that chat-template boundaries are unchanged.
    """
    ids = tokenizer(full_user_text, add_special_tokens=False).input_ids
    details = []
    for token in tokens:
        token_ids = tokenizer(token, add_special_tokens=False).input_ids
        roundtrip = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == token
        details.append({'token': token, 'ids': token_ids, 'roundtrip': roundtrip,
                        'present_in_user_ids': contiguous(ids, token_ids)})
    return details
