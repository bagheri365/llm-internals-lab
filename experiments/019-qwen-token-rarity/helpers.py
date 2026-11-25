"""Pure utilities for a balanced contextual-likelihood study."""
import math
import random
import unicodedata


def token_class(text):
    if text.isascii():
        if text.strip().isalpha():
            return 'latin_letters'
        if text.strip().isdigit():
            return 'digits'
        return 'ascii_other'
    return 'non_ascii'


def usable(tokenizer, token_id):
    if token_id in set(tokenizer.all_special_ids):
        return None
    s = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not s or '\ufffd' in s or any(unicodedata.category(ch).startswith('C') for ch in s):
        return None
    if tokenizer(s, add_special_tokens=False).input_ids != [token_id]:
        return None
    return s


def strata(records, per_bin, seed):
    """Quantile bins within token class; balance bins by downsampling."""
    if per_bin < 1:
        raise ValueError('per_bin must be positive')
    rng = random.Random(seed)
    result = []
    by_class = {}
    for row in records:
        by_class.setdefault(row['class'], []).append(row)
    for cls, group in sorted(by_class.items()):
        group.sort(key=lambda r: (-r['logprob'], r['token_id']))
        if len(group) < 8:
            continue
        bins = [group[(i*len(group))//4:((i+1)*len(group))//4] for i in range(4)]
        n = min(per_bin, *(len(b) for b in bins))
        for j, b in enumerate(bins):
            chosen = rng.sample(b, n)
            for item in chosen:
                result.append({**item, 'likelihood_bin': j})
    return result


def summarize(rows):
    groups = {}
    for r in rows:
        k = (r['class'], r['likelihood_bin'])
        groups.setdefault(k, []).append(r)
    return [{'class': k[0], 'likelihood_bin': k[1], 'n': len(v),
             'mean_delta': sum(x['activation_delta'] for x in v)/len(v),
             'min_delta': min(x['activation_delta'] for x in v),
             'mean_abs_delta': sum(abs(x['activation_delta']) for x in v)/len(v),
             'mean_margin_change': sum(x['margin_change'] for x in v)/len(v)}
            for k,v in sorted(groups.items())]
