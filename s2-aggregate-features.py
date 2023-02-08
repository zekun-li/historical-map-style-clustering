import sys
import os 
import pickle
import argparse 
import numpy as np 
import pdb 

def aggregate(args):
    file_list = os.listdir(args.input_dir)
    ret_dict = {}
    for file_path in file_list:
        external_id = file_path.split('.pkl')[0].split('embedding_')[1]
        file_path = os.path.join(args.input_dir, file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        embed = np.mean(data['embed'], axis = 0)
        ret_dict[external_id] = embed
        
    with open(args.output_path, 'wb') as f:
        pickle.dump(ret_dict, f)    


def main():

    parser = argparse.ArgumentParser()
    

    parser.add_argument('--input_dir',type=str, default = '/data2/rumsey_clustering/all-maps/')
    parser.add_argument('--output_path', type=str, default='/data2/rumsey_clustering/all-maps-feat.pkl')
    
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    aggregate(args)

if __name__ == '__main__':
    main()


