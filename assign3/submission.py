import shell
import util
import wordsegUtil



############################################################
# Problem 1: Word Segmentation

# Problem 1a: Solve the word segmentation problem under a unigram model

class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state) == 0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        l = []
        for i in range(len(state)):
            p = (state[:i+1],state[i+1:],self.unigramCost(state[:i+1]))
            l.append(p)
        return l
        # END_YOUR_ANSWER

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch()
    ucs.solve(WordSegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return " ".join(ucs.actions if ucs.actions else [])
    # END_YOUR_ANSWER

# Problem 1b: Solve the k-word segmentation problem under a unigram model

class KWordSegmentationProblem(util.SearchProblem):
    def __init__(self, k, query, unigramCost):
        self.k = k
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (self.k,self.query)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state[1]) == 0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
        l = []
        k, s = state
        if k>0:
            for i in range(len(s)):
                p = (s[:i+1],(k-1,s[i+1:]),self.unigramCost(s[:i+1]))
                l.append(p)
        return l
        # END_YOUR_ANSWER

def segmentKWords(k, query, unigramCost):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(KWordSegmentationProblem(k,query,unigramCost))
    return " ".join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 2: Vowel Insertion

# Problem 2a: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return tuple([wordsegUtil.SENTENCE_BEGIN]+self.queryWords)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state) <= 1
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)
        l = list()
        actions = set()
        actions.add(state[1])
        for i in self.possibleFills(state[1]):
            actions.add(i)
        for i in actions:
            p = (i, (i,)+state[2:], self.bigramCost(state[0],i))
            l.append(p)
        return l
        # END_YOUR_ANSWER

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(VowelInsertionProblem(queryWords,bigramCost,possibleFills))
    return " ".join(ucs.actions if ucs.actions else [])
    # END_YOUR_ANSWER

# Problem 2b: Solve the limited vowel insertion problem under a bigram cost

class LimitedVowelInsertionProblem(util.SearchProblem):
    def __init__(self, impossibleVowels, queryWords, bigramCost, possibleFills):
        self.impossibleVowels = impossibleVowels
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return tuple([wordsegUtil.SENTENCE_BEGIN]+self.queryWords)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state) <= 1
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 10 lines of code, but don't worry if you deviate from this)
        l = list()
        actions = list()
        for i in self.possibleFills(state[1]):
            if (set(i)-set(state[1]))&set(self.impossibleVowels) == set():
                actions.append(i)
        if len(actions) == 0:
            actions.append(state[1])
        for i in actions:
            l.append((i, (i,)+state[2:], self.bigramCost(state[0],i)))
        return l
        # END_YOUR_ANSWER

def insertLimitedVowels(impossibleVowels, queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(LimitedVowelInsertionProblem(impossibleVowels,queryWords,bigramCost,possibleFills))
    return " ".join(ucs.actions if ucs.actions else [])
    # END_YOUR_ANSWER

############################################################
# Problem 3: Putting It Together

# Problem 3a: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (wordsegUtil.SENTENCE_BEGIN, self.query)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state[1]) == 0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        prev_word, str_left = state
        l = []
        for i in range(len(str_left)):
            selected = str_left[:i+1]
            for p_word in list(self.possibleFills(selected))+[selected]:
                reg = (p_word, (p_word, str_left[i+1:]), self.bigramCost(prev_word,p_word))
                l.append(reg)
        return l
        # END_YOUR_ANSWER

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(JointSegmentationInsertionProblem(query,bigramCost,possibleFills))
    return " ".join(ucs.actions if ucs.actions else [])
    # END_YOUR_ANSWER

############################################################
# Problem 4: A* search

# Problem 4a: Define an admissible but not consistent heuristic function

class SimpleProblem(util.SearchProblem):
    def __init__(self):
        # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
        self.start = "a"
        self.end = "e"
        # END_YOUR_ANSWER

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.start
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == self.end
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
        d = {"a":[("a","b",1),("a","c",1)],"b":[("b","d",1)],"c":[("c","d",4)],"d":[("d","e",17)]}
        return d[state]
        # END_YOUR_ANSWER

def admissibleButInconsistentHeuristic(state):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    h = {"a":8,"b":7,"c":3,"d":0,"e":0}
    return h[state]
    # END_YOUR_ANSWER

# Problem 4b: Apply a heuristic function to the joint segmentation-and-insertion problem

def makeWordCost(bigramCost, wordPairs):
    """
    :param bigramCost: learned bigram cost from a training corpus
    :param wordPairs: all word pairs in the training corpus
    :returns: wordCost, which is a function from word to cost
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    cache = {pair[1]:bigramCost(pair[0],pair[1]) for pair in wordPairs}
    def u(w):
        return cache[w] if w in cache else bigramCost(wordsegUtil.SENTENCE_UNK,w)
    return u
    # END_YOUR_ANSWER

class RelaxedProblem(util.SearchProblem):
    def __init__(self, query, wordCost, possibleFills):
        self.query = query
        self.wordCost = wordCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state) == 0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        l = []
        for i in range(len(state)):
            if self.possibleFills(state[:i+1]):
                l.append(min([(p_word, state[i+1:], self.wordCost(p_word)) for p_word in list(self.possibleFills(state[:i+1]))+[state[:i+1]]],key=lambda x:x[2]))
        return l
        # END_YOUR_ANSWER

def makeHeuristic(query, wordCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    dp = util.DynamicProgramming(RelaxedProblem(query, wordCost, possibleFills))
    return lambda x: dp(x[1])
    # END_YOUR_ANSWER

def fastSegmentAndInsert(query, bigramCost, wordCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills),makeHeuristic(query,wordCost,possibleFills))
    return " ".join(ucs.actions if ucs.actions else [])
    # END_YOUR_ANSWER

############################################################

if __name__ == '__main__':
    shell.main()
