
import argparse
from utils import read_jsonl, write_jsonl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()

    data = read_jsonl(args.input_path)
    n = args.n

    partition_size = int(len(data) / n + 1)
    j = 0
    for i in range(0, len(data), partition_size):
        print(f'writing part {j}')
        save_path = args.input_path.replace('.jsonl', f'_part_{j}.jsonl')
        write_jsonl(data[i:i+partition_size], save_path,'w')
        j+=1
