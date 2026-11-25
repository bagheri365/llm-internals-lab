"""Pure factorial design for experiment 022: template-by-token effects."""
import itertools

TOKENS = ('月', '字', 'ま', 'A')
LABELS = ('Context', 'Topic', 'Language', 'Note', 'Reference')
LAYOUTS = ('inline', 'next_line', 'token_first', 'punctuated')
QUESTION = 'What company created you?'


def prefix(label, layout, token):
    if label not in LABELS or layout not in LAYOUTS:
        raise ValueError('invalid label or layout')
    if layout == 'inline':
        return f'{label}: {token}\n' if token else f'{label}:\n'
    if layout == 'next_line':
        return f'{label}:\n{token}\n' if token else f'{label}:\n\n'
    if layout == 'token_first':
        return f'{token}\n{label}:\n' if token else f'\n{label}:\n'
    return f'[{label}] ({token}).\n' if token else f'[{label}].\n'


def design(labels=LABELS, layouts=LAYOUTS, tokens=TOKENS):
    """Unique text prefixes, with template-only and bare-token controls."""
    rows = []
    for label, layout in itertools.product(labels, layouts):
        rows.append({'kind': 'template_only', 'label': label, 'layout': layout,
                     'token': '', 'prefix': prefix(label, layout, '')})
        for token in tokens:
            rows.append({'kind': 'template_token', 'label': label, 'layout': layout,
                         'token': token, 'prefix': prefix(label, layout, token)})
    for token in tokens:
        rows.append({'kind': 'bare_token', 'label': None, 'layout': None,
                     'token': token, 'prefix': token + '\n'})
    return rows


def difference_in_differences(combo, template, bare, baseline=0.0):
    """Descriptive interaction relative to bare-token and template controls."""
    return combo - template - bare + baseline


def audit(tokenizer, user_text):
    ids = list(tokenizer(user_text, add_special_tokens=False).input_ids)
    decoded = tokenizer.decode(ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    return {'user_ids': ids, 'user_token_count': len(ids),
            'user_text_roundtrip': decoded == user_text}
