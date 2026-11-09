import re
_SPAN_VALUE = '(?:"[^"]*"|\'[^\']*\'|[^\\s,<>]+)'
_EB_SPAN_RESIDUAL = re.compile(
    '<(?P<empty_tag>eb\\d*)\\s*>\\s*</(?P=empty_tag)\\s*>(?P<attrs>(?:[\\s,]+(?:rowspan|colspan)\\s*=\\s*'
     + _SPAN_VALUE + ')+)[\\s,]*>[\\s,]*</td\\s*>', re.I)

def _decomma_tag(text):
    quote, result = None, []
    for char in text:
        if char in ('"', "'"):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        result.append(' ' if char == ',' and quote is None else char)
    return ''.join(result)


def normalize_empty_cell_span_encoding(raw):

    def restore(match):
        return '<td' + _decomma_tag(match.group('attrs')) + '></td>'
    return _EB_SPAN_RESIDUAL.subn(restore, raw)



def normalize_structure_for_teds(raw):
    if not isinstance(raw, str):
        raise TypeError('Structure must be a string')
    text, _ = normalize_empty_cell_span_encoding(raw)
    text = re.sub('<(?P<empty>eb\\d*)\\s*>\\s*</(?P=empty)\\s*>',
        '<td></td>', text, flags=re.I)
    text = re.sub('<[^>]*>', lambda m: '<' + _decomma_tag(m.group()[
        1:-1]) + '>', text)
    return re.sub('(?<=>)[\\s,]+(?=<)', '', text)
