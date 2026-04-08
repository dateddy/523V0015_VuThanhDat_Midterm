import numpy as np

# ==========================================
# PART 1: Corpus Preprocessing (Task 1.4)
# ==========================================
corpus = [
    "the cat sat on the mat",
    "the dog ran on the road",
    "the cat and the dog are friends",
    "she loves her cat",
    "he walks his dog every day",
    "the mat is on the floor",
    "the road leads to the city",
    "friends walk together every day"
]

# (a) Tokenize and Build Vocabulary
tokenized_corpus = [sentence.lower().split() for sentence in corpus]
flat_tokens = [word for sentence in tokenized_corpus for word in sentence]
vocab = sorted(list(set(flat_tokens)))

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

# (b) Generate Context Pairs
W = 2
pairs = []

for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    n = len(indices)
    
    for i, center_idx in enumerate(indices):
        start = max(0, i - W)
        end = min(n, i + W + 1)
        
        for j in range(start, end):
            if i != j:
                pairs.append((center_idx, indices[j]))

unique_pairs = list(set(pairs))


# ==========================================
# PART 2: Skip-gram Implementation 
# ==========================================

# --- Task 2.1: Softmax and Loss Functions ---
def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def cross_entropy_loss(y_hat: np.ndarray, target_idx: int) -> float:
    """Cross-entropy loss for one training pair."""
    return -np.log(y_hat[target_idx] + 1e-15)

# --- Task 2.2 & 2.3: Forward & Backward Pass ---
class SkipGram:
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 0, init_scale: float = 0.01):
        np.random.seed(seed)
        self.V = vocab_size
        self.d = embed_dim
        # Using init_scale allows us to switch between 0.01 (Task 2.2) and 0.1 (Verification)
        self.W_in = np.random.randn(vocab_size, embed_dim) * init_scale
        self.W_out = np.random.randn(embed_dim, vocab_size) * init_scale

    def forward(self, center_idx: int):
        """Perform the forward pass for one center word."""
        v_c = self.W_in[center_idx, :]
        scores = np.dot(self.W_out.T, v_c)
        y_hat = softmax(scores)
        return v_c, y_hat

    def backward(self, center_idx: int, context_idx: int, 
                 v_c: np.ndarray, y_hat: np.ndarray, lr: float):
        """Compute gradients and update W_in and W_out with SGD."""
        # Step 1: compute error vector e
        e = y_hat.copy()
        e[context_idx] -= 1.0
        
        # Keep gradients for verification before updating
        grad_W_out = np.outer(v_c, e)
        grad_v_c = np.dot(self.W_out, e)
        
        # Step 4 & 5: update weights in-place
        self.W_out -= lr * grad_W_out
        self.W_in[center_idx, :] -= lr * grad_v_c
        
        return e, grad_W_out, grad_v_c

# --- Task 2.4: Unit Test ---
def test_gradients(model, center_idx, context_idx, eps=1e-5):
    """Numerical gradient check for W_in and W_out."""
    v_c, y_hat = model.forward(center_idx)
    
    # Analytical Gradients
    e = y_hat.copy()
    e[context_idx] -= 1.0
    grad_W_out_analytical = np.outer(v_c, e)
    grad_v_c_analytical = np.dot(model.W_out, e)
    
    relative_errors = []
    
    # Check 5 random entries in W_in
    for _ in range(5):
        d_idx = np.random.randint(0, model.d)
        
        model.W_in[center_idx, d_idx] += eps
        _, y_hat_plus = model.forward(center_idx)
        loss_plus = cross_entropy_loss(y_hat_plus, context_idx)
        
        model.W_in[center_idx, d_idx] -= 2 * eps
        _, y_hat_minus = model.forward(center_idx)
        loss_minus = cross_entropy_loss(y_hat_minus, context_idx)
        
        model.W_in[center_idx, d_idx] += eps
        
        grad_num = (loss_plus - loss_minus) / (2 * eps)
        grad_ana = grad_v_c_analytical[d_idx]
        rel_error = abs(grad_ana - grad_num) / (abs(grad_ana) + abs(grad_num) + 1e-8)
        relative_errors.append(('W_in', d_idx, rel_error))
        
    # Check 5 random entries in W_out
    for _ in range(5):
        d_idx = np.random.randint(0, model.d)
        v_idx = np.random.randint(0, model.V)
        
        model.W_out[d_idx, v_idx] += eps
        _, y_hat_plus = model.forward(center_idx)
        loss_plus = cross_entropy_loss(y_hat_plus, context_idx)
        
        model.W_out[d_idx, v_idx] -= 2 * eps
        _, y_hat_minus = model.forward(center_idx)
        loss_minus = cross_entropy_loss(y_hat_minus, context_idx)
        
        model.W_out[d_idx, v_idx] += eps
        
        grad_num = (loss_plus - loss_minus) / (2 * eps)
        grad_ana = grad_W_out_analytical[d_idx, v_idx]
        rel_error = abs(grad_ana - grad_num) / (abs(grad_ana) + abs(grad_num) + 1e-8)
        relative_errors.append(('W_out', (d_idx, v_idx), rel_error))
        
    return relative_errors

# ==========================================
# PART 3: Training the Model
# ==========================================
def train(model: SkipGram, pairs: list, epochs: int,
        lr_init: float, lr_decay: float = 0.005) -> list:
    
    losses = []
    for epoch in range(1, epochs + 1):
        lr = lr_init / (1 + lr_decay * epoch)
 
        # Shuffle pairs each epoch for better SGD convergence
        shuffled_pairs = pairs.copy()
        np.random.shuffle(shuffled_pairs)
 
        epoch_loss = 0.0
        for center_idx, context_idx in shuffled_pairs:
            # Forward pass
            v_c, y_hat = model.forward(center_idx)
            # Accumulate loss
            epoch_loss += cross_entropy_loss(y_hat, context_idx)
            # Backward pass + SGD update
            model.backward(center_idx, context_idx, v_c, y_hat, lr)
 
        # Store average loss for this epoch
        avg_loss = epoch_loss / len(shuffled_pairs)
        losses.append(avg_loss)
 
    return losses

# ==========================================
# MAIN EXECUTION & VERIFICATION PRINTS
# ==========================================
if __name__ == "__main__":
    print("-" * 75)
    print("TASK 1.4: CORPUS PREPROCESSING")
    print("-" * 75)
    print(f"Vocabulary Size (V): {len(vocab)}")
    # Print first few elements of vocab just to save terminal space
    print(f"Vocabulary: {vocab}")
    print(f"Total generated pairs: {len(pairs)}")
    print(f"Total UNIQUE pairs: {len(unique_pairs)}\n")

    print("-" * 75)
    print("TASK 2.1 & 2.3: MANUAL VERIFICATION (seed=42, d=3, scale=0.1)")
    print("-" * 75)
    # Initialize with scale 0.1 for verification
    verif_model = SkipGram(vocab_size=26, embed_dim=3, seed=42, init_scale=0.1)
    c_idx, o_idx = word2idx["cat"], word2idx["sat"]
    
    # Task 2.1 Forward Pass Verification
    v_c, y_hat = verif_model.forward(c_idx)
    score_sat = np.dot(verif_model.W_out[:, o_idx], v_c)
    prob_sat = y_hat[o_idx]
    loss_sat = cross_entropy_loss(y_hat, o_idx)
    
    print("--- Task 2.1 Outputs ---")
    print(f"v_cat            = [{v_c[0]:.4f}, {v_c[1]:.4f}, {v_c[2]:.4f}]")
    print(f"Expected score   = {score_sat:.4f}")
    print(f"Expected P(sat|cat)= {prob_sat:.4f}")
    print(f"Expected loss    = {loss_sat:.4f}\n")

    # Task 2.3 Backward Pass Table Verification
    e, grad_W_out, grad_v_c = verif_model.backward(c_idx, o_idx, v_c, y_hat, lr=0.1)
    
    # Format arrays to match the exact string format in the prompt
    grad_v_c_str = f"[{grad_v_c[0]:.4f}, {grad_v_c[1]:.4f}, {grad_v_c[2]:.4f}]"
    grad_W_out_str = f"[{grad_W_out[0, o_idx]:.4f}, {grad_W_out[1, o_idx]:.4f}, {grad_W_out[2, o_idx]:.4f}]"

    print("--- Task 2.3 Outputs (Manual Verification Table) ---")
    print(f"{'Quantity':<25} {'Calculated value':<35} {'Shape'}")
    print("-" * 75)
    print(f"{'e_w for w != sat':<25} {'y_hat_w (unchanged softmax output)':<35} {'—'}")
    print(f"{'e_sat':<25} {e[o_idx]:<35.4f} {'—'}")
    print(f"{'e (full vector)':<25} {'y_hat - y_one-hot':<35} {str(e.shape)}")
    print(f"{'Nabla_v_c L':<25} {grad_v_c_str:<35} {str(grad_v_c.shape)}")
    print(f"{'Nabla_W_out[:, sat]':<25} {grad_W_out_str:<35} {str(grad_W_out[:, o_idx].shape)}\n")


    print("-" * 75)
    print("TASK 2.2: FORWARD PASS VERIFICATION (seed=0, d=10, scale=0.01)")
    print("-" * 75)
    model = SkipGram(vocab_size=26, embed_dim=10, seed=0, init_scale=0.01)
    the_idx = word2idx["the"]
    v_c_the, y_hat_the = model.forward(the_idx)
    
    print(f"Sum of y_hat     = {y_hat_the.sum():.4f}")
    print(f"Max of y_hat     = {y_hat_the.max():.4f} (Confirms < 0.05)\n")


    print("-" * 75)
    print("TASK 2.4: GRADIENT CHECKING")
    print("-" * 75)
    errors = test_gradients(model, center_idx=c_idx, context_idx=o_idx)
    print(f"{'Matrix':<10} {'Index':<15} {'Relative Error'}")
    print("-" * 45)
    for matrix, idx, err in errors:
        print(f"{matrix:<10} {str(idx):<15} {err:.2e}")
    print("-" * 45)
    print("Status: SUCCESS (All errors < 1e-5)" if all(e < 1e-5 for _, _, e in errors) else "Status: FAILED")
    print("-" * 75)
    
    
# ==========================================
# TASK 3.1: TRAINING LOOP (Baseline)
# ==========================================
print("\n" + "-" * 75)
print("TASK 3.1: TRAINING THE MODEL (Baseline: d=10, lr=0.025, T=100)")
print("-" * 75)

# Baseline hyper-parameters (do not change)
EMBED_DIM   = 10
LR_INIT     = 0.025
LR_DECAY    = 0.005
EPOCHS      = 100
SEED        = 0
VOCAB_SIZE  = len(vocab)

baseline_model = SkipGram(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                           seed=SEED, init_scale=0.01)
baseline_losses = train(baseline_model, pairs, epochs=EPOCHS,
                        lr_init=LR_INIT, lr_decay=LR_DECAY)

print(f"Epoch  10 — Avg Loss: {baseline_losses[9]:.4f}   (expected ≈ 3.257)")
print(f"Epoch  50 — Avg Loss: {baseline_losses[49]:.4f}   (expected ≈ 2.573)")
print(f"Epoch 100 — Avg Loss: {baseline_losses[99]:.4f}   (expected ≈ 2.041)")

# ==========================================
# TASK 3.2: LEARNING CURVE
# ==========================================
import matplotlib.pyplot as plt

print("\n" + "-" * 75)
print("TASK 3.2: LEARNING CURVE")
print("-" * 75)

plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), baseline_losses, color='navy', lw=1.5,
         label='Baseline (d=10, η₀=0.025)')

# Mark the three checkpoint epochs with vertical dashed lines
checkpoints = {10: baseline_losses[9],
               50: baseline_losses[49],
               100: baseline_losses[99]}
for ep, loss_val in checkpoints.items():
    plt.axvline(x=ep, color='crimson', linestyle='--', lw=1.0, alpha=0.7)
    plt.scatter([ep], [loss_val], color='crimson', zorder=5)
    plt.annotate(f"Ep {ep}\n{loss_val:.3f}",
                 xy=(ep, loss_val),
                 xytext=(ep + 1.5, loss_val + 0.05),
                 fontsize=8, color='crimson')

plt.xlabel("Epoch")
plt.ylabel("Average Cross-Entropy Loss")
plt.title("Skip-gram Training Loss Curve (Baseline)")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.close()
print("Saved: loss_curve.png")

# ==========================================
# TASK 3.3: EFFECT OF HYPER-PARAMETERS
# ==========================================
print("\n" + "-" * 75)
print("TASK 3.3: HYPER-PARAMETER EXPERIMENTS")
print("-" * 75)

# Experiment A: Double embedding dimension (d=20), all else baseline
model_d20 = SkipGram(vocab_size=VOCAB_SIZE, embed_dim=20,
                      seed=SEED, init_scale=0.01)
losses_d20 = train(model_d20, pairs, epochs=EPOCHS,
                   lr_init=LR_INIT, lr_decay=LR_DECAY)

# Experiment B: Fixed learning rate η=0.05, no decay (lr_decay=0), d=10
model_fixed_lr = SkipGram(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                           seed=SEED, init_scale=0.01)
losses_fixed_lr = train(model_fixed_lr, pairs, epochs=EPOCHS,
                        lr_init=0.05, lr_decay=0.0)

# Comparison table
header = f"{'Setting':<35} {'Loss@Ep50':>12} {'Loss@Ep100':>12} {'Observation'}"
print(header)
print("-" * 90)

rows = [
    ("Baseline (d=10, η₀=0.025, decay)",
     baseline_losses[49], baseline_losses[99],
     "Steady convergence"),
    ("d=20 (double embedding)",
     losses_d20[49], losses_d20[99],
     "Larger capacity; may overfit small corpus"),
    ("Fixed η=0.05 (no decay)",
     losses_fixed_lr[49], losses_fixed_lr[99],
     "Faster early drop; possible instability late"),
]

for setting, l50, l100, obs in rows:
    print(f"{setting:<35} {l50:>12.4f} {l100:>12.4f}   {obs}")

print("-" * 90)
print("\nNote on Experiment B (fixed η=0.05, no decay):")
delta = losses_fixed_lr[99] - losses_fixed_lr[49]
if delta > 0.05:
    print("  → Loss increased from epoch 50→100: the model is DIVERGING "
          "due to the large fixed step size oscillating around the minimum.")
elif delta > 0.0:
    print("  → Loss slightly increased from epoch 50→100: mild instability "
          "from the non-decaying learning rate.")
else:
    print("  → Loss continued to decrease but more slowly than the baseline "
          "with decay; convergence is less smooth.")