
import argparse
from utils import read_jsonl, write_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_list", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    input_path_list = args.input_path_list.split(',')
    data = []
    for p in input_path_list:
        file_name = p.split('/')[-1]
        print(f'reading {file_name}')
        data.extend(read_jsonl(p))
    print(f'writing to {args.output_path}')
    write_jsonl(data, args.output_path,'w')