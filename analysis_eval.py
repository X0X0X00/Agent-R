
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_env_scores(directory):
    """
    Analyzes the env_score from JSON files in the given directory.
    """
    env_scores = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    env_scores.append(data['env_score'])
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {filename}: {e}")
                continue

    if not env_scores:
        print("No env_scores found.")
        return

    env_scores = np.array(env_scores)
    
    # 打印env_score的数量
    num_env_scores = len(env_scores)
    print(f"Number of env_scores: {num_env_scores}")

    # 分析数据分布
    min_env_score = np.min(env_scores)
    max_env_score = np.max(env_scores)
    mean_env_score = np.mean(env_scores)
    median_env_score = np.median(env_scores)
    std_env_score = np.std(env_scores)

    print(f"Minimum env_score: {min_env_score}")
    print(f"Maximum env_score: {max_env_score}")
    print(f"Mean env_score: {mean_env_score}")
    print(f"Median env_score: {median_env_score}")
    print(f"Standard deviation of env_score: {std_env_score}")

    # 绘制直方图
    try:
        plt.hist(env_scores, bins=20)  # You can adjust the number of bins
        plt.xlabel("Env Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Env Scores")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error plotting histogram: {e}")
        print("Please make sure matplotlib is installed. You can install it using 'pip install matplotlib'")

def main():
    parser = argparse.ArgumentParser(description="Analyze env_score from JSON files in a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory containing JSON files.")
    args = parser.parse_args()

    analyze_env_scores(args.directory)

if __name__ == "__main__":
    main()
