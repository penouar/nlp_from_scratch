import numpy as np


def load_embeddings(file_path, max_words=50000):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)
        for i, line in enumerate(f):
            if i >= max_words: break
            parts = line.rstrip().split(' ')
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    return embeddings

def load_dictionary(file_path):
    dictionary = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            src_word, tgt_word = line.rstrip().split()
            dictionary.append((src_word, tgt_word))
    return dictionary

def create_matrices(dictionary, src_embeddings, tgt_embeddings):
    X, Y = [], []
    valid_pairs = []
    for src_word, tgt_word in dictionary:
        if src_word in src_embeddings and tgt_word in tgt_embeddings:
            X.append(src_embeddings[src_word])
            Y.append(tgt_embeddings[tgt_word])
            valid_pairs.append((src_word, tgt_word))
    return np.array(X), np.array(Y), valid_pairs


def compute_loss(X, W_T, Y):
    diff = np.dot(X, W_T) - Y
    return np.linalg.norm(diff, ord='fro') ** 2

def compute_gradient(X, W_T, Y):
    diff = np.dot(X, W_T) - Y
    return 2 * np.dot(X.T, diff)

def gradient_descent(X, Y, alpha=0.0001, max_iter=500):
    d = X.shape[1]
    W_T = np.random.randn(d, d) * 0.01

    losses = []

    for i in range(max_iter):
        loss = compute_loss(X, W_T, Y)
        losses.append(loss)

        grad = compute_gradient(X, W_T, Y)

        W_T = W_T - alpha * grad

        if i >= 10:
            loss_diff = losses[-11] - losses[-1]
            if loss_diff > 0 and loss_diff < 1e-4:
                print(f"Converged at iteration {i} with loss {loss:.4f}")
                break

        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return W_T


def translate_word(word, src_embeddings, tgt_embeddings, W_T, top_k=5):
    if word not in src_embeddings:
        return []

    src_vec = src_embeddings[word]
    translated_vec = np.dot(src_vec, W_T)

    similarities = []
    tgt_words = list(tgt_embeddings.keys())
    tgt_matrix = np.array(list(tgt_embeddings.values()))

    norms_tgt = np.linalg.norm(tgt_matrix, axis=1)
    norm_translated = np.linalg.norm(translated_vec)

    dot_products = np.dot(tgt_matrix, translated_vec)
    cos_sim = dot_products / (norms_tgt * norm_translated + 1e-10)

    top_indices = np.argsort(cos_sim)[-top_k:][::-1]

    return [tgt_words[i] for i in top_indices]

def evaluate_accuracy(test_pairs, src_embeddings, tgt_embeddings, W_T):
    top1_correct = 0
    top5_correct = 0
    total = len(test_pairs)

    print("Evaluating test set... This might take a moment.")
    for i, (src_word, true_tgt_word) in enumerate(test_pairs):
        predictions = translate_word(src_word, src_embeddings, tgt_embeddings, W_T, top_k=5)

        if len(predictions) == 0:
            total -= 1
            continue

        if true_tgt_word == predictions[0]:
            top1_correct += 1
        if true_tgt_word in predictions:
            top5_correct += 1

    top1_acc = (top1_correct / total) * 100 if total > 0 else 0
    top5_acc = (top5_correct / total) * 100 if total > 0 else 0

    return top1_acc, top5_acc

if __name__ == "__main__":
    try:
        print("1. Loading Data...")
        src_emb = load_embeddings('cc.cs.300.vec', max_words=50000)
        tgt_emb = load_embeddings('cc.en.300.vec', max_words=50000)

        train_dict = load_dictionary('cs-en.train.txt')
        test_dict = load_dictionary('cs-en.test.txt')

        print("2. Creating Matrices...")
        X_train, Y_train, _ = create_matrices(train_dict, src_emb, tgt_emb)
        _, _, test_pairs_valid = create_matrices(test_dict, src_emb, tgt_emb)

        print(f"Training shape: X={X_train.shape}, Y={Y_train.shape}")

        print("3. Starting Gradient Descent...")
        W_T = gradient_descent(X_train, Y_train, alpha=0.00005, max_iter=1000)

        print("\n4. Translation Example:")
        sample_word = "pes"
        if sample_word in src_emb:
            translations = translate_word(sample_word, src_emb, tgt_emb, W_T)
            print(f"Top 5 translations for '{sample_word}': {translations}")

        print("\n5. Evaluating Accuracy...")
        top1, top5 = evaluate_accuracy(test_pairs_valid[:500], src_emb, tgt_emb, W_T)
        print(f"Top-1 Accuracy: {top1:.2f}%")
        print(f"Top-5 Accuracy: {top5:.2f}%")

    except FileNotFoundError:
        print("Error: Could not find the embedding or dictionary files.")
        print("Please download the .vec and .txt files and ensure the file paths in the code match.")
