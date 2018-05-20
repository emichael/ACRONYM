#!/usr/bin/env python2


import argparse
import collections
import hashlib
import os
import random
import re
import string
import time

import cPickle as pickle

import arxiv
import tqdm


INDEX_FOLDER = 'indices'
BATCH_SIZE = 100
CATEGORIES = {
    'cs.AI': "Artificial Intelligence",
    'cs.CL': "Computation and Language",
    'cs.CC': "Computational Complexity",
    'cs.CE': "Computational Engineering, Finance, and Science",
    'cs.CG': "Computational Geometry",
    'cs.GT': "Computer Science and Game Theory",
    'cs.CV': "Computer Vision and Pattern Recognition",
    'cs.CY': "Computers and Society",
    'cs.CR': "Cryptography and Security",
    'cs.DS': "Data Structures and Algorithms",
    'cs.DB': "Databases",
    'cs.DL': "Digital Libraries",
    'cs.DM': "Discrete Mathematics",
    'cs.DC': "Distributed, Parallel, and Cluster Computing",
    'cs.ET': "Emerging Technologies",
    'cs.FL': "Formal Languages and Automata Theory",
    'cs.GL': "General Literature",
    'cs.GR': "Graphics",
    'cs.AR': "Hardware Architecture",
    'cs.HC': "Human-Computer Interaction",
    'cs.IR': "Information Retrieval",
    'cs.IT': "Information Theory",
    'cs.LG': "Learning",
    'cs.LO': "Logic in Computer Science",
    'cs.MS': "Mathematical Software",
    'cs.MA': "Multiagent Systems",
    'cs.MM': "Multimedia",
    'cs.NI': "Networking and Internet Architecture",
    'cs.NE': "Neural and Evolutionary Computing",
    'cs.NA': "Numerical Analysis",
    'cs.OS': "Operating Systems",
    'cs.OH': "Other Computer Science",
    'cs.PF': "Performance",
    'cs.PL': "Programming Languages",
    'cs.RO': "Robotics",
    'cs.SI': "Social and Information Networks",
    'cs.SE': "Software Engineering",
    'cs.SD': "Sound",
    'cs.SC': "Symbolic Computation",
    'cs.SY': "Systems and Control"
}


class Index(object):
    def __init__(self, query=None, max_results=None):
        if query is None:
            query = ''
        if max_results is None:
            max_results=1000

        self.query = query
        self.max_results = max_results

        # maps letters to word frequencies
        self.letter_map = collections.defaultdict(collections.Counter)

        # maps lowercase acronym to (acronym, expansion) tuples
        self.acronyms = collections.defaultdict(set)

        # counts frequencies of pairs of words TODO: lowercase
        self.word_pair_map = collections.defaultdict(collections.Counter)

    # Utils for saving and loading to a file
    @property
    def _file_name(self):
        return os.path.join(
            INDEX_FOLDER, '%s.index' % hashlib.md5(
             '%s#%s' % (self.query, self.max_results)).hexdigest())

    def save(self):
        if not os.path.exists(INDEX_FOLDER):
            os.mkdir(INDEX_FOLDER)
        pickle.dump(self, open(self._file_name, 'wb'))

    def already_saved(self):
        """Return whether or not this index already exists on disk."""
        return os.path.isfile(self._file_name)

    def load(self):
        """Return a new Index from file."""
        return pickle.load(open(self._file_name, 'rb'))

    def _query_results(self):
        # TODO: generator?

        q_string = ' OR '.join(["cat:%s" % c for c in CATEGORIES.keys()])
        if self.query:
            q_string = "%s AND (%s)" % (self.query, q_string)

        results = []
        prev_results = []
        with tqdm.tqdm(desc="Fetching results from arXiv",
                       total=self.max_results) as pbar:
            for i in range(self.max_results / BATCH_SIZE + (
                    1 if self.max_results % BATCH_SIZE else 0)):
                start = i * BATCH_SIZE
                num = min(BATCH_SIZE, self.max_results - start)

                failed_attempts = 0
                new_results = []
                max_failed_attempts = 2
                while failed_attempts < max_failed_attempts:
                    new_results = arxiv.query(
                        search_query=q_string, start=start, max_results=num)

                    # Check to see if we found all results
                    if len(new_results) == num and new_results != prev_results:
                        prev_results = new_results
                        break

                    failed_attempts += 1
                    time.sleep(1)

                results += new_results
                pbar.update(len(new_results))

                if failed_attempts >= max_failed_attempts:
                    break

        return results

    @staticmethod
    def _words(s):
        return re.findall(r'\w+', s)

    @staticmethod
    def _one_line(s):
        return ' '.join(s.splitlines())

    @staticmethod
    def _title(result):
        return Index._one_line(result['title'])

    @staticmethod
    def _abstract(result):
        return Index._one_line(result['summary'])

    @staticmethod
    def _is_acronym(s, acr):
        if not acr or not s:
            return False

        if not acr[0].isupper():
            return False

        if acr[0] != s[0].upper():
            return False

        return True

    @staticmethod
    def _acronyms(s):
        results = []

        # First, grab all phrases in parentheses
        parens = re.finditer(r'\([^\)]+\)', s)
        parens = [(m.start(), m.group()) for m in (parens if parens else [])]

        for start, term in parens:
            term = term[1:-1] # strip the parentheses
            ws = Index._words(term)

            # First, check if the parenthetical is expansion of preceeding word
            if len(ws) > 1:
                m = re.search(r'(\w+)[^\w]+$', s[:start])
                if not m:
                    continue
                preceeding_word = m.group(1)

                if Index._is_acronym(term, preceeding_word):
                    results.append((preceeding_word, term))

            # Next, check if this word is the acronym
            if len(ws) == 1:
                acr = ws[0]
                fl = acr[0] # first letter

                # Grab the preceeding 2x words, check each of them
                preceeding_words = re.finditer(r'\w+', s[:start])
                if not preceeding_words: # TODO: is this necessary?
                    continue
                preceeding_words = list(preceeding_words)[::-1][:len(acr) * 2]

                if not preceeding_words:
                    continue

                last_word_end = (preceeding_words[0].start() +
                                 len(preceeding_words[0].group()))

                for m in preceeding_words:
                    w = m.group(0)
                    if not w or w[0].upper() != fl.upper():
                        continue

                    phrase = s[m.start():last_word_end]

                    # TODO: maybe break early or check all possibilities and
                    #       break ties with scores, potentially try to see if
                    #       unmatched letters are prefix of previous word
                    if Index._is_acronym(phrase, acr):
                        results.append((acr, phrase))

        return results

    @staticmethod
    def _all_acronyms(query_results):
        acrs = []
        for result in query_results:
            acrs += Index._acronyms(Index._title(result))
            acrs += Index._acronyms(Index._abstract(result))
        return acrs

    def _add_word_pairs(self, s):
        ws = self._words(s)

        for i, w1 in enumerate(ws):
            if i + 1 < len(ws):
                w2 = ws[i+1]
                self.word_pair_map[w1][w2] += 1

    def _add_acronym(self, acr, exp):
        # First, add the acronym to the map
        self.acronyms[acr.lower()].add((acr, exp))

        ws = self._words(exp)

        taken = 0
        for l in acr:
            for w in ws[taken:]:
                if w and w[0].upper() == l.upper():
                    self.letter_map[l][w] += 1
                    taken += 1

    def build(self):
        """Queries the arXiv and builds the Index."""
        results = self._query_results()

        for a, e in self._all_acronyms(results):
            self._add_acronym(a, e)

        for r in results:
            self._add_word_pairs(self._abstract(r))

    def find_acronyms(self, acronym):
        """Finds all instances of acronym in the data."""
        return self.acronyms[acronym.lower()]

    @staticmethod
    def _sample(counter):
        return random.sample(list(counter.elements()), 1)[0]

    @staticmethod
    def _cap_words(words):
        for i in range(len(words)):
            w = words[i]

            if w == w.lower() and (i == 0 or len(w) > 3):
                words[i] = string.capwords(w)

        return ' '.join(words)

    def gen_acronym(self, acronym):
        """Randomly generates the given acronym using the Index."""
        words = []
        previous = None

        for l in acronym:
            possibilities = self.letter_map[l]

            # Delete all the already used words
            for w in words:
                del possibilities[w]

            if not possibilities:
                continue

            if previous is not None:
                # TODO: intersect better ??? normalize and add ???
                seeded = self.word_pair_map[previous] & possibilities
                if seeded:
                    possibilities = seeded

            previous = self._sample(possibilities)
            words.append(previous)

        return self._cap_words(words)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-q', '--query', type=str, nargs='?', default=None,
                        help="the keywords to search the arXiv for")
    parser.add_argument('-r', '--max-results', type=int, nargs='?',
                        default=None,
                        help="maximum results to fetch from arXiv")
    parser.add_argument('-f', '--find', action='store_true',
                        help="finds instances of the acronym (instead of "
                             "generating it)")
    parser.add_argument('acronym', type=str, nargs='?', default=None,
                        metavar='A', help="the acronym to create")

    args = parser.parse_args()

    index = Index(query=args.query, max_results=args.max_results)
    if index.already_saved():
        index = index.load()
    else:
        index.build()
        index.save()

    if args.find:
        for acr, exp in index.find_acronyms(args.acronym):
            print acr, exp
    else:
        print index.gen_acronym(args.acronym)


if __name__ == '__main__':
    main()
