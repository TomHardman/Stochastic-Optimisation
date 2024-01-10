import os

def decode(message_file):
    with open(message_file, 'r') as f:
        lines = f.readlines() # reads lines in text file to list of strings
    
    codebook = {} # dictionary to store word-number pairs
    largest_num = 0

    for line in lines: # iterate through lines and store pairs in codebook
        num, word = line.split(' ')
        num = int(num) # convert from string to int
        word = word.split('\n')[0] # remove newline character
        codebook[num] = word
        largest_num = max([num, largest_num])
    
    line_ends = []
    next_end = 1
    n = 2

    while next_end <= largest_num: # generate list of triangular numbers
        line_ends.append(next_end)
        next_end = int((n**2 + n) / 2)
        n += 1
    
    output_arr = [codebook.get(line_end) for line_end in line_ends] # fetch message words from codebook
    output = " ".join(output_arr) # join words to single string

    return output
    

        

if __name__ == '__main__':
    decode('coding_qual_input.txt')