import os
import json
import random
from sklearn.model_selection import train_test_split
import shutil
import argparse


def split_dataset(folder_path, train_size=0.8, seed=777):
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        files = os.listdir(category_path)
        os.makedirs(os.path.join(folder_path, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'test', category), exist_ok=True)
        train_files, test_files = train_test_split(files, train_size=train_size, random_state=seed)
        for file in train_files:
            shutil.move(os.path.join(category_path, file), os.path.join(folder_path, 'train', category, file))
        for file in test_files:
            shutil.move(os.path.join(category_path, file), os.path.join(folder_path, 'test', category, file))
            

    print(f"Dataset split into 'train' and 'test' folders with {train_size*100}% training data.")


def process_dataset(folder_path, n, seed):

    category_file = os.path.join(folder_path, 'category.txt')
    test_file = os.path.join(folder_path, 'test.json')
    train_file = os.path.join(folder_path, 'train.json')

    with open(category_file, 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    
    category_map = {int(line.split(',')[0]): line.split(',')[1] for line in categories}

    random.seed(seed)

    selected_ids = sorted(random.sample(list(category_map.keys()), n))
    selected_map = {i: category_map[i] for i in selected_ids}

    print(f"Selected {n} categories (seed={seed}):")
    
    for cid in selected_ids:
        print(f"  - {cid}: {category_map[cid]}")

    with open(test_file, 'r') as f:
        test_data = json.load(f)
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    def filter_data(data, selected_ids):
        return [item for item in data if int(item.split('/')[0]) in selected_ids]

    new_test_data = filter_data(test_data, selected_ids)
    new_train_data = filter_data(train_data, selected_ids)

    with open(os.path.join(folder_path, 'new_category.txt'), 'w') as f:
        for i in selected_ids:
            f.write(f"{i},{category_map[i]}\n")

    with open(os.path.join(folder_path, 'new_test.json'), 'w') as f:
        json.dump(new_test_data, f, indent=4)

    with open(os.path.join(folder_path, 'new_train.json'), 'w') as f:
        json.dump(new_train_data, f, indent=4)

    print(f"\nZapisano do plików:")
    print(f"    - new_category.txt ({n} categories)")
    print(f"    - new_test.json     ({len(new_test_data)} samples)")
    print(f"    - new_train.json    ({len(new_train_data)} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the dataset folder.")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of the dataset to include in the train split.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducibility.")
    args = parser.parse_args()
    split_dataset(folder_path=args.folder_path, train_size=args.train_size, seed=args.seed)
