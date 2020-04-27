import json
import os

from pattern import en as patten
from stanza.server import CoreNLPClient
from tqdm import tqdm


# Map to pattern.en aliases
# http://www.clips.ua.ac.be/pages/pattern-en#conjugation
POS_TO_PATTERN = {
        'vb': 'inf',  # Infinitive
        'vbp': '1sg',  # non-3rd-person singular present
        'vbz': '3sg',  # 3rd-person singular present
        'vbg': 'part',  # gerund or present participle
        'vbd': 'p',  # past
        'vbn': 'ppart',  # past participle
}
# Tenses prioritized by likelihood of arising
PATTERN_TENSES = ['inf', '3sg', 'p', 'part', 'ppart', '1sg']


class ConstituencyParse(object):
    """A CoreNLP constituency parse (or a node in a parse tree).
    
    Word-level constituents have |word| and |index| set and no children.
    Phrase-level constituents have no |word| or |index| and have at least one child.
    """
    def __init__(self, tag, children=None, word=None, index=None):
        self.tag = tag
        if children:
            self.children = children
        else:
            self.children = None
        self.word = word
        self.index = index

    @classmethod
    def _recursive_parse_corenlp(cls, tokens, i, j):
        orig_i = i
        if tokens[i] == '(':
            tag = tokens[i + 1]
            children = []
            i = i + 2
            while True:
                child, i, j = cls._recursive_parse_corenlp(tokens, i, j)
                if isinstance(child, cls):
                    children.append(child)
                    if tokens[i] == ')': 
                        return cls(tag, children), i + 1, j
                else:
                    if tokens[i] != ')':
                        raise ValueError('Expected ")" following leaf')
                    return cls(tag, word=child, index=j), i + 1, j + 1
        else:
            # Only other possibility is it's a word
            return tokens[i], i + 1, j

    @classmethod
    def from_corenlp(cls, s):
        """Parses the "parse" attribute returned by CoreNLP parse annotator."""
        # "parse": "(ROOT\n  (SBARQ\n    (WHNP (WDT What)\n      (NP (NN portion)\n        (PP (IN                       of)\n          (NP\n            (NP (NNS households))\n            (PP (IN in)\n              (NP (NNP             Jacksonville)))))))\n    (SQ\n      (VP (VBP have)\n        (NP (RB only) (CD one) (NN person))))\n    (. ?        )))",
        s_spaced = s.replace('\n', ' ').replace('(', ' ( ').replace(')', ' ) ')
        tokens = [t for t in s_spaced.split(' ') if t]
        tree, index, num_words = cls._recursive_parse_corenlp(tokens, 0, 0)
        if index != len(tokens):
            raise ValueError('Only parsed %d of %d tokens' % (index, len(tokens)))
        return tree

    def is_singleton(self):
        if self.word: return True
        if len(self.children) > 1: return False
        return self.children[0].is_singleton()
        
    def print_tree(self, indent=0):
        spaces = '  ' * indent
        if self.word:
            print(('%s%s: %s (%d)' % (spaces, self.tag, self.word, self.index)).encode('utf-8'))
        else:
            print('%s%s:' % (spaces, self.tag))
            for c in self.children:
                c.print_tree(indent=indent + 1)

    def get_phrase(self):
        if self.word: return self.word
        toks = []
        for i, c in enumerate(self.children):
            p = c.get_phrase()
            if i == 0 or p.startswith("'"):
                toks.append(p)
            else:
                toks.append(' ' + p)
        return ''.join(toks)

    def get_start_index(self):
        if self.index is not None: return self.index
        return self.children[0].get_start_index()

    def get_end_index(self):
        if self.index is not None: return self.index + 1
        return self.children[-1].get_end_index()

    @classmethod
    def _recursive_replace_words(cls, tree, new_words, i):
        if tree.word:
            new_word = new_words[i]
            return (cls(tree.tag, word=new_word, index=tree.index), i + 1)
        new_children = []
        for c in tree.children:
            new_child, i = cls._recursive_replace_words(c, new_words, i)
            new_children.append(new_child)
        return cls(tree.tag, children=new_children), i

    @classmethod
    def replace_words(cls, tree, new_words):
        """Return a new tree, with new words replacing old ones."""
        new_tree, i = cls._recursive_replace_words(tree, new_words, 0)
        if i != len(new_words):
            raise ValueError('len(new_words) == %d != i == %d' % (len(new_words), i))
        return new_tree


def read_const_parse(parse_str):
    tree = ConstituencyParse.from_corenlp(parse_str)
    new_tree = compress_whnp(tree)
    return new_tree


def compress_whnp(tree, inside_whnp=False):
    if not tree.children:
        return tree
    for i, c in enumerate(tree.children):
        tree.children[i] = compress_whnp(c, inside_whnp=inside_whnp or tree.tag == 'WHNP')
    if tree.tag != 'WHNP':
        if inside_whnp:
            return ConstituencyParse('NP', children=[tree])
        return tree
    wh_word = None
    new_np_children = []
    new_siblings = []
    for i, c in enumerate(tree.children):
        if i == 0:
            if c.tag in ('WHNP', 'WHADJP', 'WHAVP', 'WHPP'):
                wh_word = c.children[0]
                new_np_children.extend(c.children[1:])
            elif c.tag in ('WDT', 'WP', 'WP$', 'WRB'):
                wh_word = c
            else:
                return tree
        else:
            if c.tag == 'SQ':
                new_siblings = tree.children[i:]
                break
            new_np_children.append(ConstituencyParse('NP', children=[c]))
    if new_np_children:
        new_np = ConstituencyParse('NP', children=new_np_children)
        new_tree = ConstituencyParse('WHNP', children=[wh_word, new_np])
    else:
        new_tree = tree
    if new_siblings:
        new_tree = ConstituencyParse('SBARQ', children=[new_tree] + new_siblings)
    return new_tree


class ConversionRule(object):
    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        raise NotImplementedError


class ConstituencyRule(ConversionRule):
    """A rule for converting question to sentence based on constituency parse."""

    def __init__(self, in_pattern, out_pattern, postproc=None):
        self.in_pattern = in_pattern
        self.out_pattern = str(out_pattern)
        self.name = in_pattern
        if postproc:
            self.postproc = postproc
        else:
            self.postproc = {}

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        pattern_toks = self.in_pattern.split(' ')
        match = match_pattern(self.in_pattern, const_parse)
        appended_clause = False
        if not match:
            appended_clause = True
            new_pattern = '$PP , ' + self.in_pattern
            pattern_toks = new_pattern.split(' ')
            match = match_pattern(new_pattern, const_parse)
        if not match:
            new_pattern = '$SBAR , ' + self.in_pattern
            pattern_toks = new_pattern.split(' ')
            match = match_pattern(new_pattern, const_parse)
        if not match:
            return None
        appended_clause_match = None
        fmt_args = [a]
        for t, m in zip(pattern_toks, match):
            if t.startswith('$') or '/' in t:
                phrase = convert_whp(m, q, a, tokens)
                if not phrase:
                    phrase = m.get_phrase()
                fmt_args.append(phrase)
        if appended_clause:
            appended_clause_match = fmt_args[1]
            fmt_args = [a] + fmt_args[2:]
        for i in range(len(fmt_args)):
            if i in self.postproc:
                fmt_args[i] = run_postprocessing(fmt_args[i], self.postproc[i], fmt_args)
        output = self.gen_output(fmt_args)
        if appended_clause:
            output = appended_clause_match + ', ' + output
        if run_fix_style:
            output = fix_style(output)
        return output

    def gen_output(self, fmt_args):
        """By default, use self.out_pattern.  Can be overridden."""
        return self.out_pattern.format(*fmt_args)


def run_postprocessing(s, rules, all_args):
    rule_list = rules.split(',')
    for rule in rule_list:
        if rule == 'lower':
            s = s.lower()
        elif rule.startswith('tense-'):
            ind = int(rule[6:])
            orig_vb = all_args[ind]
            tenses = patten.tenses(orig_vb)
            for tense in PATTERN_TENSES:  # Prioritize by PATTERN_TENSES
                if tense in tenses:
                    break
            else:  # Default to first tense
                tense = PATTERN_TENSES[0]
            s = patten.conjugate(s, tense)
        elif rule in POS_TO_PATTERN:
            s = patten.conjugate(s, POS_TO_PATTERN[rule])
    return s


class FindWHPRule(ConversionRule):
    """A rule that looks for $WHP's from right to left and does replacements."""
    name = 'FindWHP'

    def _recursive_convert(self, node, q, a, tokens, found_whp):
        if node.word:
            return node.word, found_whp
        if not found_whp:
            whp_phrase = convert_whp(node, q, a, tokens)
            if whp_phrase:
                return whp_phrase, True
        child_phrases = []
        for c in node.children[::-1]:
            c_phrase, found_whp = self._recursive_convert(c, q, a, tokens, found_whp)
            child_phrases.append(c_phrase)
        out_toks = []
        for i, p in enumerate(child_phrases[::-1]):
            if i == 0 or p.startswith("'"):
                out_toks.append(p)
            else:
                out_toks.append(' ' + p)
        return ''.join(out_toks), found_whp

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        out_phrase, found_whp = self._recursive_convert(const_parse, q, a, tokens, False)
        if found_whp:
            if run_fix_style:
                out_phrase = fix_style(out_phrase)
            return out_phrase
        return None


CONVERSION_RULES = [
    ConstituencyRule('$WHP:what $Be $NP called that $VP', '{2} that {3} {1} called {1}'),
    ConstituencyRule('how $JJ $Be $NP $IN $NP', '{3} {2} {0} {1} {4} {5}'),
    ConstituencyRule('how $JJ $Be $NP $SBAR', '{3} {2} {0} {1} {4}'),
    ConstituencyRule('how $JJ $Be $NP', '{3} {2} {0} {1}'),
    ConstituencyRule('$WHP:when/where $Do $NP', '{3} occurred in {1}'),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb', '{3} {4} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP/$PP', '{3} {4} {5} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP $PP', '{3} {4} {5} {6} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Be $NP', '{3} {2} in {1}'),
    ConstituencyRule('$WHP:when/where $Verb $NP $VP/$ADJP', '{3} {2} {4} in {1}'),
    ConstituencyRule("$WHP:what/which/who $Do $NP do", '{3} {1}', {0: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb", '{3} {4} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $IN/$NP", '{3} {4} {5} {1}', {4: 'tense-2', 0: 'vbg'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $PP", '{3} {4} {1} {5}', {4: 'tense-2', 0: 'vbg'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $NP $VP", '{3} {4} {5} {6} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB", '{3} {4} to {5} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB $VP", '{3} {4} to {5} {1} {6}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $NP $IN $VP", '{3} {4} {5} {6} {1} {7}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $PP/$S/$VP/$SBAR/$SQ", '{3} {4} {1} {5}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $PP $PP/$S/$VP/$SBAR", '{3} {4} {1} {5} {6}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP of $NP $Verb/$Part $IN", '{3} of {4} {2} {5} {6} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $NP $IN", '{3} {2} {4} {5} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $VP/$IN", '{3} {2} {4} {1}'),
    ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $IN $NP/$VP", '{1} {2} {3} {4} {5}'),
    ConstituencyRule('$WHP:what/which/who $Be/$MD $NP $Verb $PP', '{3} {2} {4} {1} {5}'),
    ConstituencyRule('$WHP:what/which/who $Be/$MD $NP/$VP/$PP', '{1} {2} {3}'),
    ConstituencyRule("$WHP:how $Be/$MD $NP $VP", '{3} {2} {4} by {1}'),
    ConstituencyRule("$WHP:what/which/who $VP", '{1} {2}'),
    ConstituencyRule('$IN what/which $NP $Do $NP $Verb $NP', '{5} {6} {7} {1} the {3} of {0}',
                     {1: 'lower', 6: 'tense-4'}),
    ConstituencyRule('$IN what/which $NP $Be $NP $VP/$ADJP', '{5} {4} {6} {1} the {3} of {0}',
                     {1: 'lower'}),
    ConstituencyRule('$IN what/which $NP $Verb $NP/$ADJP $VP', '{5} {4} {6} {1} the {3} of {0}',
                     {1: 'lower'}),
    FindWHPRule(),
]


def fix_style(s):
    """Minor, general style fixes for questions."""
    s = s.replace('?', '')
    s = s.strip(' .')
    if s[0] == s[0].lower():
        s = s[0].upper() + s[1:]
    return s + '.'


def match_pattern(pattern, const_parse):
    pattern_toks = pattern.split(' ')
    whole_phrase = const_parse.get_phrase()
    if whole_phrase.endswith('?') or whole_phrase.endswith('.'):
        pattern_toks.append(whole_phrase[-1])
    matches = []
    success = _recursive_match_pattern(pattern_toks, [const_parse], matches)
    if success:
        return matches
    else:
        return None


def _check_match(node, pattern_tok):
    if pattern_tok in CONST_PARSE_MACROS:
        pattern_tok = CONST_PARSE_MACROS[pattern_tok]
    if ':' in pattern_tok:
        lhs, rhs = pattern_tok.split(':')
        match_lhs = _check_match(node, lhs)
        if not match_lhs:
            return False
        phrase = node.get_phrase().lower()
        retval = any(phrase.startswith(w) for w in rhs.split('/'))
        return retval
    elif '/' in pattern_tok:
        return any(_check_match(node, t) for t in pattern_tok.split('/'))
    return ((pattern_tok.startswith('$') and pattern_tok[1:] == node.tag) or
            (node.word and pattern_tok.lower() == node.word.lower()))


CONST_PARSE_MACROS = {
    '$Noun': '$NP/$NN/$NNS/$NNP/$NNPS',
    '$Verb': '$VB/$VBD/$VBP/$VBZ',
    '$Part': '$VBN/$VG',
    '$Be': 'is/are/was/were',
    '$Do': "do/did/does/don't/didn't/doesn't",
    '$WHP': '$WHADJP/$WHADVP/$WHNP/$WHPP',
}


def convert_whp(node, q, a, tokens):
    if node.tag in ('WHNP', 'WHADJP', 'WHADVP', 'WHPP'):
        cur_phrase = node.get_phrase()
        cur_tokens = tokens[node.get_start_index():node.get_end_index()]
        for r in WHP_RULES:
            phrase = r.convert(cur_phrase, a, cur_tokens, node, run_fix_style=False)
            if phrase:
                return phrase
    return None


def _recursive_match_pattern(pattern_toks, stack, matches):
    """Recursively try to match a pattern, greedily."""
    if len(matches) == len(pattern_toks):
        return len(stack) == 0
    if len(stack) == 0:
        return False
    cur_tok = pattern_toks[len(matches)]
    node = stack.pop()
    is_match = _check_match(node, cur_tok)
    if is_match:
        cur_num_matches = len(matches)
        matches.append(node)
        new_stack = list(stack)
        success = _recursive_match_pattern(pattern_toks, new_stack, matches)
        if success:
            return True
        while len(matches) > cur_num_matches:
            matches.pop()
    if not node.children:
        return False
    stack.extend(node.children[::-1])
    return _recursive_match_pattern(pattern_toks, stack, matches)


class ReplaceRule(ConversionRule):
    """A simple rule that replaces some tokens with the answer."""

    def __init__(self, target, replacement='{}', start=False):
        self.target = target
        self.replacement = str(replacement)
        self.name = 'replace(%s)' % target
        self.start = start

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        t_toks = self.target.split(' ')
        q_toks = q.rstrip('?.').split(' ')
        replacement_text = self.replacement.format(a)
        for i in range(len(q_toks)):
            if self.start and i != 0:
                continue
            if ' '.join(q_toks[i:i + len(t_toks)]).rstrip(',').lower() == self.target:
                begin = q_toks[:i]
                end = q_toks[i + len(t_toks):]
                output = ' '.join(begin + [replacement_text] + end)
                if run_fix_style:
                    output = fix_style(output)
                return output
        return None


class AnswerRule(ConversionRule):
    """Just return the answer."""
    name = 'AnswerRule'

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        return a


WHP_RULES = [
    ConstituencyRule('$IN what/which type/sort/kind/group of $NP/$Noun', '{1} {0} {4}'),
    ConstituencyRule('$IN what/which type/sort/kind/group of $NP/$Noun $PP', '{1} {0} {4} {5}'),
    ConstituencyRule('$IN what/which $NP', '{1} the {3} of {0}'),
    ConstituencyRule('$IN $WP/$WDT', '{1} {0}'),
    ConstituencyRule('what/which type/sort/kind/group of $NP/$Noun', '{0} {3}'),
    ConstituencyRule('what/which type/sort/kind/group of $NP/$Noun $PP', '{0} {3} {4}'),
    ConstituencyRule('what/which $NP', 'the {2} of {0}'),
    ConstituencyRule('how many/much $NP', '{0} {2}'),
    ReplaceRule('what'),
    ReplaceRule('who'),
    ReplaceRule('how many'),
    ReplaceRule('how much'),
    ReplaceRule('which'),
    ReplaceRule('where'),
    ReplaceRule('when'),
    ReplaceRule('why'),
    ReplaceRule('how'),
    AnswerRule(),
]


def run_conversion(qas, corenlp_home):
    os.environ['CORENLP_HOME'] = 'stanford-corenlp-full-2018-10-05'
    ret = list()
    with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse'], timeout=30000, memory='16G', properties={'ssplit.eolonly': True, 'ssplit.newlineIsSentenceBreak': 'always', 'outputFormat':'json'}, endpoint='http://localhost:9001') as client:
        for question, answer in tqdm(qas):
            parse = client.annotate(question)['sentences'][0]
            tokens = parse['tokens']
            const_parse = read_const_parse(parse['parse'])
            for rule in CONVERSION_RULES:
                sent = rule.convert(question, answer, tokens, const_parse)
                if sent:
                    ret.append([question, answer, sent])
                    break
            else:
                ret.append([question, answer, None])
    return ret
    


def main():
    qas = [["What is the current series where the new series began in June 2011?", "CB\u00b706\u00b7ZZ"], ["What is the format for South Australia?", "Snnn\u00b7aaa"]]
    sents = run_conversion(qas, 'stanford-corenlp-full-2018-10-05')
    print(sents)


if __name__ == '__main__':
    main()
