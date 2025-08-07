import os
import json
import random

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
    process_dataset(folder_path=".", n=7, seed=777)
