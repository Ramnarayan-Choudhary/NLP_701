import json
import argparse
from pathlib import Path
from collections import defaultdict
import random
from typing import List, Dict, Tuple


def load_dataset(file_path: str) -> List[Dict]:
    data = []
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data


def stratified_split(data: List[Dict], train_ratio: float, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    random.seed(seed)
    stratified_groups = defaultdict(list)
    for example in data:
        validity = bool(example['validity'])
        plausibility = bool(example['plausibility'])
        key = (validity, plausibility)
        stratified_groups[key].append(example)
    print(f"{'Validity':<12} {'Plausibility':<15} {'Count':<10} {'Percentage'}")
    total = len(data)
    for (validity, plausibility), examples in sorted(stratified_groups.items()):
        count = len(examples)
        percentage = (count / total) * 100
        print(f"{str(validity):<12} {str(plausibility):<15} {count:<10} {percentage:.2f}%")
    
    train_data = []
    test_data = []
    
    for (validity, plausibility), examples in sorted(stratified_groups.items()):
        random.shuffle(examples)
        n_train = max(1, int(len(examples) * train_ratio)) 
        group_train = examples[:n_train]
        group_test = examples[n_train:]
        
        train_data.extend(group_train)
        test_data.extend(group_test)
        
        group_name = f"V={validity}, P={plausibility}"
        print(f"{group_name:<30} {len(examples):<8} {len(group_train):<8} {len(group_test):<8}")
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    return train_data, test_data


def save_dataset(data: List[Dict], file_path: str, format: str = 'json'):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    if format == 'jsonl':
        with open(file_path, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps(example) + '\n')
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data)} examples to {file_path}")


def verify_distribution(original: List[Dict], train: List[Dict], test: List[Dict]):
    def get_distribution(data):
        dist = defaultdict(int)
        for example in data:
            key = (bool(example['validity']), bool(example['plausibility']))
            dist[key] += 1
        return dist
    
    orig_dist = get_distribution(original)
    train_dist = get_distribution(train)
    test_dist = get_distribution(test)

    
    for key in sorted(orig_dist.keys()):
        validity, plausibility = key
        orig_pct = (orig_dist[key] / len(original)) * 100
        train_pct = (train_dist[key] / len(train)) * 100 if train else 0
        test_pct = (test_dist[key] / len(test)) * 100 if test else 0
        
        group_name = f"V={validity}, P={plausibility}"
        print(f"{group_name:<30} {orig_pct:<15.2f} {train_pct:<15.2f} {test_pct:<15.2f}")
    
    print("-" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="split data"
    )
    parser.add_argument(
        "input_file",
        type=str
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data_splits",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=['json', 'jsonl'],
        default='json',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
    )
    
    args = parser.parse_args()
    
    if not 0 < args.train_split < 1:
        raise ValueError(f"train_split must be between 0 and 1, got {args.train_split}")
    
    data = load_dataset(args.input_file)
    
    train_data, test_data = stratified_split(data, args.train_split, args.seed)
    verify_distribution(data, train_data, test_data)
    input_path = Path(args.input_file)
    base_name = input_path.stem
    extension = '.jsonl' if args.output_format == 'jsonl' else '.json'
    suffix = f"_{args.suffix}" if args.suffix else ""
    train_file = f"{base_name}_train{suffix}{extension}"
    test_file = f"{base_name}_test{suffix}{extension}"
    train_path = Path(args.output_dir) / train_file
    test_path = Path(args.output_dir) / test_file
    save_dataset(train_data, str(train_path), args.output_format)
    save_dataset(test_data, str(test_path), args.output_format)


if __name__ == "__main__":
    main()
