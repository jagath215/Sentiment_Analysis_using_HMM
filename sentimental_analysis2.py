import numpy as np
from Data_processor import Data_processor


possible_states = ["O", "B-positive", "I-positive", "B-neutral", "I-neutral", "B-negative", "I-negative"]


def emis_prob(state, word, training_data, emis_dict):
    if (state, word) in emis_dict.keys():
        return emis_dict[(state, word)]
    else:
        count_emission = 0
        count_state = 1
        count_word = 0

        for tweet in training_data:
            for j in range(len(tweet)):
                if tweet[j].split(" ")[0] == word:
                    count_word += 1
                if tweet[j].split(" ")[1] == state:
                    count_state += 1
                    if tweet[j].split(" ")[0] == word:
                        count_emission += 1

        if count_word == 0:
            result = float(1/count_state)
        else:
            result = float(count_emission/count_state)

        emis_dict[(state, word)] = result
        return result


def trans_prob(state1, state2, training_data, trans_dict):
    if (state1, state2) in trans_dict.keys():
        return trans_dict[(state1, state2)]
    else:
        count_transition = 0
        count_state1 = 0
        
        for tweet in training_data:
            for j in range(len(tweet)-1):
                if len(tweet[j].split(" ")) > 1 and tweet[j].split(" ")[1] == state1:
                    count_state1 += 1
                    if tweet[j+1].split(" ")[1] == state2:
                        count_transition += 1

        result = float(count_transition/count_state1)
        trans_dict[(state1, state2)] = result
        return result


def viterbi_algorithm(pi, a, b, sequence, possible_states):
    n = len(possible_states)
    T = len(sequence)

    viterbi_matrix = np.zeros((n, T))
    backpointers = np.zeros((n, T), dtype=int)

    viterbi_matrix[:, 0] = pi * b[:, sequence[0]]

    for t in range(1, T):
        for j in range(n):
            viterbi_matrix[j, t] = np.max(viterbi_matrix[:, t-1] * a[:, j] * b[j, sequence[t]])
            backpointers[j, t] = np.argmax(viterbi_matrix[:, t-1] * a[:, j])

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(viterbi_matrix[:, -1])

    for t in range(T-2, -1, -1):
        path[t] = backpointers[path[t+1], t+1]

    return path


def decode_path(path, hidden_states):
    return [hidden_states[state_index] for state_index in path]


def viterbi_label(dev_datapath, training_datapath):
    trans_dict = {}
    emis_dict = {}

    training_data = Data_processor(training_datapath).data
    input_data = Data_processor(dev_datapath).data
    total_tweets_no = len(input_data)
    
    outpath = './Output/output.out'
    outfile = open(outpath, 'w', encoding='utf8')

    for tweet_no in range(total_tweets_no):
        sequence = input_data[tweet_no]
        pi = np.ones(len(possible_states)) / len(possible_states)  # Initial probabilities
        a = np.zeros((len(possible_states), len(possible_states)))  # Transition probabilities
        b = np.zeros((len(possible_states), len(sequence)))  # Emission probabilities

        # Populate transition probabilities matrix
        for i, state1 in enumerate(possible_states):
            for j, state2 in enumerate(possible_states):
                a[i, j] = trans_prob(state1, state2, training_data, trans_dict)

        # Populate emission probabilities matrix
        for i, state in enumerate(possible_states):
            for j, word in enumerate(sequence):
                b[i, j] = emis_prob(state, word, training_data, emis_dict)

        # Run Viterbi algorithm
        opt_y_seq = viterbi_algorithm(pi, a, b, list(range(len(sequence))), possible_states)

        # Decode the path
        decoded_sequence = decode_path(opt_y_seq, possible_states)

        # Write the results to the output file
        for word, tag in zip(sequence, decoded_sequence):
            output = word + " " + tag + "\n"
            outfile.write(output)

        outfile.write("\n")
        print(str(tweet_no + 1) + "/" + str(total_tweets_no) + " done")


    print("Labeling completed!")
    outfile.close()



viterbi_label('./Input/input2.in', './train/train.txt')
