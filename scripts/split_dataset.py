import os
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

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the dataset folder.")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of the dataset to include in the train split.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducibility.")
    args = parser.parse_args()
    split_dataset(folder_path=args.folder_path, train_size=args.train_size, seed=args.seed)
