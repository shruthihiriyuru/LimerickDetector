#!/usr/bin/env python
import argparse
import sys
import codecs
if sys.version_info[0] == 2:
  from itertools import izip
else:
  izip = zip
from collections import defaultdict as dd
import re
import os.path
import gzip
import tempfile
import shutil
import atexit

# Use word_tokenize to split raw text into words
from string import punctuation

import nltk
from nltk.tokenize import word_tokenize

scriptdir = os.path.dirname(os.path.abspath(__file__))

reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')

def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code if code.endswith("t") else code+"t") if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

def addonoffarg(parser, arg, dest=None, default=True, help="TODO"):
  ''' add the switches --arg and --no-arg that set parser.arg to true/false, respectively'''
  group = parser.add_mutually_exclusive_group()
  dest = arg if dest is None else dest
  group.add_argument('--%s' % arg, dest=dest, action='store_true', default=default, help=help)
  group.add_argument('--no-%s' % arg, dest=dest, action='store_false', default=default, help="See --%s" % arg)



class LimerickDetector:

    def __init__(self):
        """
        Initializes the object to have a pronunciation dictionary available
        """
        self._pronunciations = nltk.corpus.cmudict.dict()

    def num_syllables(self, word):
        """
        Returns the number of syllables in a word.  If there's more than one
        pronunciation, take the shorter one.  If there is no entry in the
        dictionary, return 1.
        """

        pronunciations = self._pronunciations.get(word)

        if pronunciations is None:
            return 1

        syllable_count = []
        for pronunciation in pronunciations:
            count = 0
            for phoneme in pronunciation:
                phoneme = phoneme.encode('ascii', 'ignore')
                if phoneme[0] in 'AEIOU':
                    count = count + 1
            syllable_count.append(count)

        return min(syllable_count)

    def strip_sounds(self, pronunciations):

        stripped_pronunciations = []
        for pronunciation in pronunciations:
            for idx, phoneme in enumerate(pronunciation):
                phoneme = phoneme.encode('ascii', 'ignore')
                if phoneme[0] in 'AEIOU':
                    stripped_pronunciations.append(pronunciation[idx:])
                    break
        return stripped_pronunciations

    def rhymes(self, a, b):
        """
        Returns True if two words (represented as lower-case strings) rhyme,
        False otherwise.
        """

        pronunciations_a = self._pronunciations.get(a)
        pronunciations_b = self._pronunciations.get(b)

        list_a = self.strip_sounds(pronunciations_a)
        list_b = self.strip_sounds(pronunciations_b)

        for a in list_a:
            for b in list_b:
                len_a = len(a)
                len_b = len(b)
                if len_a == len_b:
                    if a == b:
                        return True
                elif len_a > len_b:
                    if a[len_a-len_b:] == b:
                        return True
                else:
                    if b[len_b-len_a:] == a:
                        return True

        return False

    def get_line_syllable_count(self, line_words):
        syllable_count = 0
        for word in line_words:
            syllable_count += self.num_syllables(word)

        return syllable_count

    def remove_punctuations(self, line_words):
        tokens = [word for word in line_words if word not in punctuation and word != "``" and word != "''"]
        return tokens

    def is_limerick(self, text):
        """
        Takes text where lines are separated by newline characters.  Returns
        True if the text is a limerick, False otherwise.

        A limerick is defined as a poem with the form AABBA, where the A lines
        rhyme with each other, the B lines rhyme with each other, and the A lines do not
        rhyme with the B lines.


        Additionally, the following syllable constraints should be observed:
          * No two A lines should differ in their number of syllables by more than two.
          * The B lines should differ in their number of syllables by no more than two.
          * Each of the B lines should have fewer syllables than each of the A lines.
          * No line should have fewer than 4 syllables

        (English professors may disagree with this definition, but that's what
        we're using here.)


        """

        all_lines = text.splitlines()
        lines = [line for line in all_lines if line.strip() != '']
        num_of_lines = len(lines)
        if num_of_lines != 5:
            return False

        # Check if first two lines rhyme
        line1_words = word_tokenize(lines[0])
        line1_words = self.remove_punctuations(line1_words)

        line2_words = word_tokenize(lines[1])
        line2_words = self.remove_punctuations(line2_words)

        if not self.rhymes(line1_words[len(line1_words)-1], line2_words[len(line2_words)-1]):
            return False

        # Check if third and fourth lines rhyme
        line3_words = word_tokenize(lines[2])
        line3_words = self.remove_punctuations(line3_words)

        line4_words = word_tokenize(lines[3])
        line4_words = self.remove_punctuations(line4_words)
        if not self.rhymes(line3_words[len(line3_words)-1], line4_words[len(line4_words)-1]):
            return False

        # Check if first and fifth or second and fifth lines rhyme
        line5_words = word_tokenize(lines[4])
        line5_words = self.remove_punctuations(line5_words)
        if not self.rhymes(line1_words[len(line1_words)-1], line5_words[len(line5_words)-1])\
                and not self.rhymes(line2_words[len(line2_words)-1], line5_words[len(line5_words)-1]):
            return False

        # Check that first and third lines do not rhyme
        if self.rhymes(line1_words[len(line1_words)-1], line3_words[len(line3_words)-1]):
            return False

        '''
          * No two A lines should differ in their number of syllables by more than two.
          * The B lines should differ in their number of syllables by no more than two.
          * Each of the B lines should have fewer syllables than each of the A lines.
          * No line should have fewer than 4 syllables
        '''
        syllable_count = []
        syllable_count.append(self.get_line_syllable_count(line1_words))
        syllable_count.append(self.get_line_syllable_count(line2_words))
        syllable_count.append(self.get_line_syllable_count(line3_words))
        syllable_count.append(self.get_line_syllable_count(line4_words))
        syllable_count.append(self.get_line_syllable_count(line5_words))

        if True in [t < 4 for t in syllable_count]:
            return False

        a_syllable_max = max([syllable_count[0], syllable_count[1], syllable_count[4]])
        a_syllable_min = min([syllable_count[0], syllable_count[1], syllable_count[4]])

        if (a_syllable_max - a_syllable_min) > 2:
            return False

        if abs(syllable_count[2] - syllable_count[3]) > 2:
            return False

        return True

    def apostrophe_tokenize(self, line):
        if "'" not in line:
            return word_tokenize(line)
        else:
            # Remove all punctuations other than apostrophe
            line = re.sub('[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', "", line)
            tokens = line.split()
            return tokens

    def guess_syllables(self, word):

        word = word.lower()
        matches = re.findall(r'[aeiou]+', word)

        count = len(matches)
        # Usually if words end with e, it won't count towards a syllable
        # Eg. : Farce, Terse
        # except when the ending e is preceded by l
        # Eg. : Able, tickle

        match = re.search(r'[^aeioul]e$', word, re.M|re.I)
        if match:
            count -= 1

        # If a word has y preceded by consonant, it counts as a syllable
        matches = re.findall(r'[^aeiou]y.*', word)
        count += len(matches)

        # For words with single e at the end, count can go to 0
        # Eg: she
        return max(1, count)

# The code below should not need to be modified
def main():
  parser = argparse.ArgumentParser(description="limerick detector. Given a file containing a poem, indicate whether that poem is a limerick or not",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input file")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")




  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))

  infile = prepfile(args.infile, 'r')
  outfile = prepfile(args.outfile, 'w')

  ld = LimerickDetector()
  lines = ''.join(infile.readlines())
  outfile.write("{}\n-----------\n{}\n".format(lines.strip(), ld.is_limerick(lines)))

  print ld.num_syllables("vile")
  print ld.guess_syllables("vile")

if __name__ == '__main__':
  main()
