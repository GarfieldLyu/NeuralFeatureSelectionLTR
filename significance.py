from scipy import stats
import argparse 
import json
from pathlib import Path
import numpy as np


""" Compute the significance of two results. """

def signific(probs1: np.array, probs2: np.array):
    """ if probs1 is greater than probs2 significantly."""
    return stats.ttest_rel(probs1, probs2, alternative='greater')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--DIRECTORY',  type=str, help='Your project directory.')
    ap.add_argument('--Dataset', type=str, help='MQ2008, MSLR30, Yahoo.')
    ap.add_argument('--Fold', type=str, default='Fold1', help='which fold?, within data_dir')
    ap.add_argument('--file_prob1', type=str, help='.json file.')
    ap.add_argument('--file_prob2', type=str)
    args = ap.parse_args()

    file_prob1 = Path(args.DIRECTORY) / 'Trained' / args.Dataset / args.Fold / args.file_prob1
    file_prob2 = Path(args.DIRECTORY) / 'Trained' / args.Dataset / args.Fold / args.file_prob2
    
    with open(file_prob1, 'r')as f:
        prob1 = json.load(f)
    
    with open(file_prob2, 'r')as f:
        prob2 = json.load(f)

    print(f'AVG file 1: {prob1[-1]}\n ')
    print(f'AVG file 2: {prob2[-1]}\n ')
    prob1 = prob1[:-1]  #the last element is the avg
    prob2 = prob2[:-1]


    for i in range(len(prob1[0])):
        p1 = [P[i] for P in prob1]
        p2 = [P[i] for P in prob2]
        print(len(p1), len(p2))
        print(signific(p1, p2))


if __name__ == '__main__':
    main()

