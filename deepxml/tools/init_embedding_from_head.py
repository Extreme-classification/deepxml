# Replace fasttext embeddings with head embeddings wherever possible

import numpy as np
import sys


def main():
    original_embeddings = np.load(sys.argv[1])
    head_embeddings = np.load(sys.argv[2])
    head_features = np.loadtxt(sys.argv[3], dtype=np.int)
    original_embeddings[head_features] = head_embeddings
    print(original_embeddings.shape, sys.argv[4])
    np.save(sys.argv[4], original_embeddings)


if __name__ == "__main__":
    main()
