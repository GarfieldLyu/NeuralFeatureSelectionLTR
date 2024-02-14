import pickle
import data_utils
import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import h5py
from tqdm import tqdm


 # split the original Istella train.txt to train and valid.txt, we din't use istella dataset in our experiments.
def split_Istella(ratio: float=0.25, seed: int=100):  
    fold_dir = Path('your-project-dir/Datasets/Istella/Fold1/')
    all_file = fold_dir / 'train_all.txt'
    with open(all_file, 'r')as f:
        lines = f.readlines()
    train, valid = train_test_split(lines, test_size=ratio, random_state=seed)
    print(f'Train size: {len(train)}, Valid size: {len(valid)}.')
    train_file = fold_dir / 'train.txt'
    valid_file = fold_dir / 'valid.txt'
    with open(train_file, 'w')as f:
        f.writelines(train)
    with open(valid_file, 'w')as f:
        f.writelines(valid)
    print('Done!')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--DIRECTORY', default='/home/lyu/LTR/', type=str, help='project directory')
    ap.add_argument('--Dataset', type=str, help='MQ2008, MSLR30.')
    ap.add_argument('--Fold', type=str, default='Fold1', help='which fold?, within data_dir')
    ap.add_argument('--task', type=str, help='h5, clean, pairs') 
    ap.add_argument('--max_size', type=int, default=20, help='maximum sample pairs') 

    args = ap.parse_args()

    if args.task == 'h5':       # save to h5py object
        for d_type in ['train', 'valid', 'test']:
            data_utils.MS_to_h5(args.DIRECTORY, args.Dataset, args.Fold, d_type)

    elif args.task == 'clean':  # remove the queries with less than 10 docs and all 0s labels
        data_path = Path(args.DIRECTORY) / "Datasets" / args.Dataset / args.Fold 
        clean(data_path)

    elif args.task == 'pairs':  # create pairs for training, only necessary for pairwise training objectives.
        data_utils.creat_pairs(args.DIRECTORY, args.Dataset, args.Fold, 'train', max_size=args.max_size)
    else:
        raise ValueError(f'Invalid task: {args.task}')
        

def clean(data_path: Path):
    """ Remove the queries with less than 10 docs and all 0s labels"""
    with open(data_path / 'train_query_group.pkl', 'rb')as f:
        group = pickle.load(f)

    train_filtered = []
    data_h5 = data_path /'train.h5'
    for index in tqdm(range(len(group))):
        if group[index] >= 10:
            start_id = sum(group[:index])
            end_id = start_id + group[index] 
            with h5py.File(data_h5, 'r')as fp:
                Ys = [fp['Y'][idx][0] for idx in range(start_id, end_id)]  # doc_len
                if len(list(set(Ys))) > 1:
                    train_filtered.append(index)
    with open(data_path / 'train_filtered.json', 'w')as f:
        json.dump(train_filtered, f)
    print(f'Saved filtered train group indexes.')
 


if __name__ == '__main__':
    #split_Istella()   # split the original Istella train.txt to train and valid.txt
    main()
