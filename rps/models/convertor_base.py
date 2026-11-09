# Adapted from MTL-TabNet, TableMASTER, and OpenMMLab components.
# Changes: rps package imports and framework-independent integration.
# Upstream notices: THIRD_PARTY_NOTICES.md and licenses/.
from rps.utils import list_from_file

class BaseConvertor:
    start_idx = end_idx = padding_idx = 0
    unknown_idx = None
    lower = False
    DICT36 = tuple('0123456789abcdefghijklmnopqrstuvwxyz')
    DICT90 = tuple(
        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]_`~'
        )

    def __init__(self, dict_type='DICT90', dict_file=None, dict_list=None):
        assert dict_type in ('DICT36', 'DICT90')
        assert dict_file is None or isinstance(dict_file, str)
        assert dict_list is None or isinstance(dict_list, list)
        self.idx2char = []
        if dict_file is not None:
            for line in list_from_file(dict_file):
                line = line.strip('\n')
                if line != '':
                    self.idx2char.append(line)
        elif dict_list is not None:
            self.idx2char = dict_list
        elif dict_type == 'DICT36':
            self.idx2char = list(self.DICT36)
        else:
            self.idx2char = list(self.DICT90)
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def num_classes(self):
        return len(self.idx2char)

    def str2idx(self, strings):
        assert isinstance(strings, list)
        indexes = []
        for string in strings:
            if self.lower:
                string = string.lower()
            index = []
            for char in string:
                char_idx = self.char2idx.get(char, self.unknown_idx)
                if char_idx is None:
                    raise Exception(
                        f'Chararcter: {char} not in dict, please check gt_label and use custom dict file, or set "with_unknown=True"'
                        )
                index.append(char_idx)
            indexes.append(index)
        return indexes

    def str2tensor(self, strings):
        raise NotImplementedError

    def idx2str(self, indexes):
        assert isinstance(indexes, list)
        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            strings.append(''.join(string))
        return strings

    def tensor2idx(self, output):
        raise NotImplementedError
