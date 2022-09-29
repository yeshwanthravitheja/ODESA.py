import numpy as np
import math

import random
from typing import List


class MorseCode:
    '''
     A markov environment which produces the given phone number in morse code as a class and other random phone numbers. 
    '''
    def __init__(self,dim,*argv):
        assert(dim == 2)
        assert(len(argv)>0)
        self.dim = dim
        # self.center = int((dim)/2)
        
        self.state_len = 5
        self.class_repr = list(argv)
        self.num_classes = len(argv)
        self.pattern_len = len(self.class_repr[0])
        self.num_patterns_per_class = 1
        self.states = {'A': '.-', 'B': '-...', 'C': '-.-.',
        'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..',
        'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---',
        'P': '.--.', 'Q': '--.-', 'R': '.-.',
        'S': '...', 'T': '-', 'U': '..-',
        'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..',
        '0': '-----', '1': '.----', '2': '..---',
        '3': '...--', '4': '....-', '5': '.....',
        '6': '-....', '7': '--...', '8': '---..',
        '9': '----.',
        }
        self.state_keys = list(self.states.keys())
        self.num_states = len(self.state_keys)

        # Add a randomo class
        # self.class_repr.append(random.choices(self.state_keys,k=self.pattern_len))

        # Select random class
        self.curr_class_idx = random.randint(0,self.num_classes-1)
        # Find the class representation of current class e.g ['Y','E','S','H']
        self.curr_class = self.class_repr[self.curr_class_idx]
        # Find the length of the current class representation 
        self.curr_class_len = len(self.curr_class)
        # Start from the first state aka letter
        self.curr_state_idx = 0
        # Find the letter representing the state e.g "Y"
        self.curr_state_key = self.curr_class[self.curr_state_idx]
        # use that key to find the morse code '-.--'
        self.curr_state = self.states[self.curr_state_key]
        # Find the length of the morse code to know when to go to next letter
        self.curr_state_len = len(self.curr_state)
        # Position within a state/letter/morsecode
        self.state_pos = 0
        # Flag to know if the last letter in the class is reached
        self.end_state = 0

        self.ts = 1

    def next_class(self):
        # Select random class
        self.curr_class_idx = random.randint(0,self.num_classes-1)

        # if random class, then change the random class pattern based on predefined random pattern len
        # if self.curr_class_idx > self.num_classes -1:
        #     self.class_repr[self.curr_class_idx] = random.choices(self.state_keys,k=self.pattern_len)
        
        # Find the class representation of current class e.g ['Y','E','S','H']
        self.curr_class = self.class_repr[self.curr_class_idx]
        # Find the length of the current class representation 
        self.curr_class_len = len(self.curr_class)

        # Start from the first state aka letter
        self.curr_state_idx = 0
        # Find the letter representing the state e.g "Y"
        self.curr_state_key = self.curr_class[self.curr_state_idx]

        # use that key to find the morse code '-.--'
        self.curr_state = self.states[self.curr_state_key]
        # Find the length of the morse code to know when to go to next letter
        self.curr_state_len = len(self.curr_state)
        # Flag to know if the last letter in the class is reached
        self.end_state = 0
        # Position within a state/letter/morsecode
        self.state_pos = 0

        self.ts += self.pattern_len*7
    
    def next_state(self):
        # Go to next letter in the word
        self.curr_state_idx += 1
        #  Check if the letter is the last letter in the word
        if self.curr_state_idx == self.curr_class_len - 1:
            # Set the flag of last letter
            self.end_state = 1

        # # Position within a state/letter/morsecode
        self.state_pos = 0
        # Find the letter representing the state e.g "Y"
        self.curr_state_key = self.curr_class[self.curr_state_idx]
        # use that key to find the morse code '-.--'
        self.curr_state = self.states[self.curr_state_key]
        # Find the length of the morse code to know when to go to next letter
        self.curr_state_len = len(self.curr_state)

        self.ts += 2


    def next_event(self):
        '''
        Returns (x,y,ts,class) - class is 1 or 0, -1 if non important event. 
        '''

        x = 0 # For all events since this is 1D problem 
        # 0 for dot and 1 for dash
        y = 0 if self.curr_state[self.state_pos] == '.' else 1
        # if random.random() > 0.9:
        #     y = random.randrange(0,self.dim)
        current_ts = self.ts * 10
        current_class = -1

        # special_state = self.special

        
        self.ts += 1

        if self.state_end():
            if self.end_state > 0:
                current_class = math.floor(self.curr_class_idx/self.num_patterns_per_class)
                # if current_class > self.num_classes -1 :
                #     current_class = -1
                self.next_class()
            else:
                self.next_state()
        else:            
            self.state_pos += 1

        return (x,y,current_ts,current_class) 


    def reset(self):

        self.curr_class_idx = random.randint(0,self.num_classes-1)
        self.curr_class = self.class_repr[self.curr_class_idx]
        self.curr_class_len = len(self.curr_class)
        self.curr_state_idx = 0
        self.curr_state_key = self.curr_class[self.curr_state_idx]
        self.curr_state = self.states[self.curr_state_key]
        self.curr_state_len = len(self.curr_state)
        self.state_pos = 0
        self.end_state = 0

        # self.state_pos = 0
        # self.curr_state = 0
        # self.end_state = 0
        self.ts = 1

        # return self.ball_center

    def state_end(self):
        return self.state_pos == self.curr_state_len - 1

