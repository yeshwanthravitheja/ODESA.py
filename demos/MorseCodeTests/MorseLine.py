import numpy as np
import math

import random
from typing import List


# poem = [
#     "shall i compare thee to a summers day",
#     "thou art more lovely and more temperate",
#     "rough winds do shake the darling buds of may",
#     "and summers lease hath all too short a date",
#     "sometime too hot the eye of heaven shines",
#     "and often is his gold complexion dimmd",
#     "and every fair from fair sometime declines",
#     "by chance or natures changing course untrimmd",
#     "but thy eternal summer shall not fade",
#     "nor lose possession of that fair thou owest",
#     "nor shall death brag thou wanderst in his shade",
#     "when in eternal lines to time thou growest",
#     "so long as men can breathe or eyes can see",
#     "so long lives this and this gives life to thee"
# ]


class MorseLine:
    '''
     A markov environment which produces the given lines in morse code and mark the each line as the specific class and other random shuffling (prob) of the line as non class. 
    '''
    def __init__(self,dim,prob,*argv):
        assert(dim == 2)
        assert(len(argv)>0)
        # assert(prob<1)
        self.dim = dim
        self.prob = prob

        # self.center = int((dim)/2)
        self.lines = []
        self.num_classes = len(argv)
        # assert(self.num_classes > 0)
        for line in argv:
            self.lines.append(line.split())
        # self.lines.append(line1.split())
        # self.lines.append(line2.split())
        self.classes = [i for i in range(self.num_classes)]
        
        # self.line = line.split()
        self.letters = {
                        "0": "-----",
                        "1": ".----",
                        "2": "..---",
                        "3": "...--",
                        "4": "....-",
                        "5": ".....",
                        "6": "-....",
                        "7": "--...",
                        "8": "---..",
                        "9": "----.",
                        "a": ".-",
                        "b": "-...",
                        "c": "-.-.",
                        "d": "-..",
                        "e": ".",
                        "f": "..-.",
                        "g": "--.",
                        "h": "....",
                        "i": "..",
                        "j": ".---",
                        "k": "-.-",
                        "l": ".-..",
                        "m": "--",
                        "n": "-.",
                        "o": "---",
                        "p": ".--.",
                        "q": "--.-",
                        "r": ".-.",
                        "s": "...",
                        "t": "-",
                        "u": "..-",
                        "v": "...-",
                        "w": ".--",
                        "x": "-..-",
                        "y": "-.--",
                        "z": "--..",
                        ".": ".-.-.-",
                        ",": "--..--",
                        "?": "..--..",
                        "!": "-.-.--",
                        "-": "-....-",
                        "/": "-..-.",
                        "@": ".--.-.",
                        "(": "-.--.",
                        ")": "-.--.-"
                        }
        self.letter_keys = list(self.letters.keys())
        self.num_states = len(self.letter_keys)
        self.curr_class = random.randint(0, self.num_classes-1)
        self.line = self.lines[self.curr_class]
        self.line_len = len(self.line)

        self.random_class = 0
        if random.random() > self.prob:
            self.curr_line = random.sample(self.line,self.line_len)
            self.random_class = 1
        else:
            self.curr_line = self.line.copy()

        self.max_word_len = 9

        self.curr_word_idx = 0
        self.curr_word = self.curr_line[self.curr_word_idx]
        self.curr_word_len = len(self.curr_word)

        self.curr_letter_idx = 0
        self.curr_letter = self.curr_word[self.curr_letter_idx]

        self.curr_state_idx = 0
        self.curr_state = self.letters[self.curr_letter]
        self.curr_state_len = len(self.curr_state)
        self.last_word = 0
        self.last_letter = 0
        self.last_state = 0

        if self.curr_state_idx == self.curr_state_len -1:
            self.last_state = 1
        if self.curr_letter_idx == self.curr_word_len -1 :
            self.last_letter = 1
        if self.curr_word_idx == self.line_len -1 :
            self.last_word = 1

        self.ts = 1
    
    def next_line(self):
        self.curr_class = random.randint(0, self.num_classes-1)
        self.line = self.lines[self.curr_class]
        self.line_len = len(self.line)
        self.random_class = 0
        if random.random() > self.prob:
            self.curr_line = random.sample(self.line,self.line_len)
            self.random_class = 1
        else:
            self.curr_line = self.line.copy()


        self.curr_word_idx = 0
        self.curr_word = self.curr_line[self.curr_word_idx]
        self.curr_word_len = len(self.curr_word)

        self.curr_letter_idx = 0
        self.curr_letter = self.curr_word[self.curr_letter_idx]

        self.curr_state_idx = 0
        self.curr_state = self.letters[self.curr_letter]
        self.curr_state_len = len(self.curr_state)
        self.last_word = 0
        self.last_letter = 0
        self.last_state = 0

        if self.curr_state_idx == self.curr_state_len -1:
            self.last_state = 1
        if self.curr_letter_idx == self.curr_word_len -1 :
            self.last_letter = 1
        if self.curr_word_idx == self.line_len -1 :
            self.last_word = 1

        self.ts += self.line_len*self.max_word_len*10*2


    def next_word(self):

        self.curr_word_idx += 1
        if self.curr_word_idx == self.line_len - 1:
            self.last_word = 1

        self.curr_word = self.curr_line[self.curr_word_idx]
        self.curr_word_len = len(self.curr_word)

        self.curr_letter_idx = 0
        self.curr_letter = self.curr_word[self.curr_letter_idx]

        self.curr_state_idx = 0
        self.curr_state = self.letters[self.curr_letter]
        self.curr_state_len = len(self.curr_state)
        self.last_letter = 0
        self.last_state = 0
        if self.curr_letter_idx == self.curr_word_len -1 :
            self.last_letter = 1
        if self.curr_state_idx == self.curr_state_len -1:
            self.last_state = 1

        self.ts += self.max_word_len*10
    
    def next_letter(self):
        self.curr_letter_idx += 1
        if self.curr_letter_idx == self.curr_word_len -1:
            self.last_letter = 1
        self.curr_letter = self.curr_word[self.curr_letter_idx]

        self.curr_state_idx = 0
        self.curr_state = self.letters[self.curr_letter]
        self.curr_state_len = len(self.curr_state)
        self.last_state = 0
        if self.curr_state_idx == self.curr_state_len -1:
            self.last_state = 1
        self.ts += 5
    
    def next_state(self):
        # Go to next letter in the word
        self.curr_state_idx += 1
        #  Check if the letter is the last letter in the word
        if self.curr_state_idx == self.curr_state_len - 1:
            # Set the flag of last letter
            self.last_state = 1
        self.ts += 1


    def next_event(self):
        '''
        Returns (x,y,ts,class) - class is 1 or 0, -1 if non important event. 
        '''

        x = 0 # For all events since this is 1D problem 
        # 0 for dot and 1 for dash
        y = 0 if self.curr_state[self.curr_state_idx] == '.' else 1
        # if random.random() > 0.9:
        #     y = random.randrange(0,self.dim)
        current_ts = self.ts * 10
        current_class = -1

        # special_state = self.special

        
        # self.ts += 1
        if self.last_state > 0:
            if self.last_letter > 0:
                if self.last_word > 0:
                    
                    if self.random_class < 1:
                        current_class = self.curr_class
                    self.next_line()
                else:
                    self.next_word()
            else:
                self.next_letter()
        else:
            self.next_state()
        


        return (x,y,current_ts,current_class,self.curr_word,self.curr_letter) 


    def reset(self):
        self.curr_class = random.randint(0, self.num_classes-1)
        self.line = self.lines[self.curr_class]
        self.line_len = len(self.line)
        self.random_class = 0
        if random.random() > self.prob:
            self.curr_line = random.sample(self.line,self.line_len)
            self.random_class = 1
        else:
            self.curr_line = self.line.copy()


        self.curr_word_idx = 0
        self.curr_word = self.curr_line[self.curr_word_idx]
        self.curr_word_len = len(self.curr_word)

        self.curr_letter_idx = 0
        self.curr_letter = self.curr_word[self.curr_letter_idx]

        self.curr_state_idx = 0
        self.curr_state = self.letters[self.curr_letter]
        self.curr_state_len = len(self.curr_state)
        self.last_word = 0
        self.last_letter = 0
        self.last_state = 0

        if self.curr_state_idx == self.curr_state_len -1:
            self.last_state = 1
        if self.curr_letter_idx == self.curr_word_len -1 :
            self.last_letter = 1
        if self.curr_word_idx == self.line_len -1 :
            self.last_word = 1

        self.ts = 1



