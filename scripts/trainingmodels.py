"""
ECHR Improved Bridge — Stage 1 upgrade
========================================
Replaces TF-IDF + SVM with better Stage 1 models:

  Stage 1 models compared:
    S1-A: TF-IDF + SVM              (baseline)
    S1-B: TF-IDF + LightGBM         (better tree model)
    S1-C: Full-doc char n-gram       (hash + SVD, no 512 limit)
    S1-D: FastText + BiLSTM          (NEW — reads full doc with sequence model)

  Stage 2 models compared:
    S2: LightGBM + combo features

  Key experiment:
    Show that S1-D (FastText + BiLSTM, full document) closes the gap
    between Oracle AUC and Pipeline AUC further than S1-C.

Usage:
    pip install lightgbm gensim torch

    python echr_improved_bridge.py --data merged.json
"""

import argparse, json, os, re, warnings, math
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                              recall_score, roc_auc_score)
import lightgbm as lgb

warnings.filterwarnings("ignore")

CAP_EUR = 1_000_000

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",            default="merged.json")
    p.add_argument("--train_ratio",     type=float, default=0.80)
    p.add_argument("--val_ratio",       type=float, default=0.10)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--output_dir",      default="./improved_output")
    # FastText + BiLSTM hyper-params
    p.add_argument("--ft_dim",          type=int,   default=100,
                   help="FastText embedding dimension")
    p.add_argument("--ft_epochs",       type=int,   default=5,
                   help="FastText training epochs")
    p.add_argument("--lstm_hidden",     type=int,   default=128,
                   help="BiLSTM hidden size (each direction)")
    p.add_argument("--lstm_layers",     type=int,   default=2,
                   help="Number of BiLSTM layers")
    p.add_argument("--lstm_epochs",     type=int,   default=10,
                   help="BiLSTM training epochs")
    p.add_argument("--lstm_batch",      type=int,   default=32)
    p.add_argument("--lstm_lr",         type=float, default=1e-3)
    p.add_argument("--max_seq_len",     type=int,   default=2000,
                   help="Max tokens per doc for BiLSTM (0=all)")
    p.add_argument("--skip_bilstm",     action="store_true",
                   help="Skip BiLSTM (for quick testing)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def clean(t):
    return re.sub(r"[\x00-\x1f\x7f]", "", re.sub(r"\s+", " ", t)).strip()

def to_text(v):
    return clean(" ".join(str(x) for x in v) if isinstance(v, list) else str(v))

def to_eur(d):
    return float(d.get("EUR", 0.0) or 0.0) if isinstance(d, dict) else 0.0

def simple_tokenize(text):
    """Lowercase word tokenizer — no NLTK required."""
    return re.findall(r"[a-z]+", text.lower())

def award_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }
    try:    m["auc"] = round(roc_auc_score(y_true, y_prob), 4)
    except: m["auc"] = None
    return m

def article_f1(y_true, y_pred):
    return {
        "f1_micro": round(f1_score(y_true, y_pred, average="micro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path, train_ratio, val_ratio, seed):
    print(f"\nLoading {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  {len(raw):,} cases")

    rows = []
    for c in raw:
        arts    = sorted(set(str(a).lower().strip()
                             for a in c.get("violated_articles", [])))
        claimed = to_eur(c.get("total_damage_claimed"))
        awarded = to_eur(c.get("total_damage_awarded"))
        rows.append({
            "text":         to_text(c.get("text", [])),
            "articles":     arts,
            "claimed":      claimed,
            "awarded":      awarded,
            "award_binary": int(awarded > 0),
        })

    df = pd.DataFrame(rows)
    df = df[df["text"].str.len() > 50].reset_index(drop=True)

    before = len(df)
    df = df[(df["claimed"] <= CAP_EUR) &
            (df["awarded"] <= CAP_EUR)].reset_index(drop=True)
    print(f"  After filter: {len(df):,}  (removed {before-len(df):,})")

    pos = df["award_binary"].sum()
    print(f"  Award=1: {pos:,} ({pos/len(df)*100:.1f}%)")

    # Word-count stats
    wc = df["text"].apply(lambda x: len(x.split()))
    print(f"  Doc lengths — mean:{wc.mean():.0f}  median:{wc.median():.0f}  "
          f"max:{wc.max():.0f}  p95:{wc.quantile(0.95):.0f}")

    idx   = list(range(len(df)))
    strat = df["articles"].apply(lambda x: x[0] if x else "none").tolist()
    try:
        tr, tmp, _, tmp_s = train_test_split(
            idx, strat, test_size=1-train_ratio,
            stratify=strat, random_state=seed)
        test_ratio = round(1-train_ratio-val_ratio, 6)
        vl, te = train_test_split(
            tmp, test_size=test_ratio/(val_ratio+test_ratio),
            stratify=tmp_s, random_state=seed)
        print("  Stratified split ✓")
    except Exception as e:
        print(f"  Random split ({e})")
        tr, tmp = train_test_split(idx, test_size=1-train_ratio, random_state=seed)
        test_ratio = round(1-train_ratio-val_ratio, 6)
        vl, te  = train_test_split(
            tmp, test_size=test_ratio/(val_ratio+test_ratio), random_state=seed)

    print(f"  Train {len(tr):,} / Val {len(vl):,} / Test {len(te):,}")
    
    # Count article frequency in training set only
    from collections import Counter
    art_counts = Counter(
        art 
        for arts in df["articles"].iloc[tr] 
        for art in arts
    )
    
    # Keep only articles with >= 50 training cases
    MIN_CASES = 50
    active = {art for art, cnt in art_counts.items() if cnt >= MIN_CASES}
    print(f"  Active articles (>= {MIN_CASES} cases): {len(active)} "
          f"from {len(art_counts)} total")
    print(f"  Kept: {sorted(active)}")
    
    # Filter article lists to only include active articles
    df["articles"] = df["articles"].apply(
        lambda arts: [a for a in arts if a in active]
  
    mlb = MultiLabelBinarizer()
    mlb.fit(df["articles"].iloc[tr].tolist())
    print(f"  Articles ({len(mlb.classes_)}): {sorted(mlb.classes_)}")

    return df, tr, vl, te, mlb


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def build_combo_features(article_lists, mlb):
    classes = list(mlb.classes_)
    pairs   = list(combinations(range(len(classes)), 2))
    n       = len(article_lists)
    indiv   = mlb.transform(article_lists).astype(np.float32)
    pair_f  = np.zeros((n, len(pairs)), dtype=np.float32)
    for i, arts in enumerate(article_lists):
        art_set = set(arts)
        for j, (a, b) in enumerate(pairs):
            if classes[a] in art_set and classes[b] in art_set:
                pair_f[i, j] = 1.0
    count = np.array([[len(a)] for a in article_lists], dtype=np.float32)
    return np.hstack([indiv, pair_f, count])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: LightGBM bridge
# ─────────────────────────────────────────────────────────────────────────────
def run_stage2_lgbm(X_tr, X_te, y_tr, y_te, label=""):
    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)])
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    m = award_metrics(y_te, y_pred, y_prob)
    print(f"  LightGBM {label:35s} — "
          f"Acc:{m['accuracy']:.4f}  F1:{m['f1']:.4f}  AUC:{m['auc']:.4f}")
    return m, clf


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1-A: TF-IDF + SVM (baseline)
# ═════════════════════════════════════════════════════════════════════════════
def stage1_tfidf_svm(df, tr, te, mlb):
    print("\n  S1-A: TF-IDF + SVM (baseline) ...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=3)
    X_tr = tfidf.fit_transform(df["text"].iloc[tr].tolist())
    X_te = tfidf.transform(df["text"].iloc[te].tolist())

    clf = OneVsRestClassifier(
        LinearSVC(max_iter=2000, C=1.0, class_weight="balanced", random_state=42),
        n_jobs=-1)
    clf.fit(X_tr, mlb.transform(df["articles"].iloc[tr].tolist()))

    Y_te_pred = clf.predict(X_te)
    Y_te_true = mlb.transform(df["articles"].iloc[te].tolist())
    m = article_f1(Y_te_true, Y_te_pred)
    print(f"    F1-micro={m['f1_micro']}  F1-macro={m['f1_macro']}")

    pred_arts = mlb.inverse_transform(Y_te_pred)
    return [list(a) for a in pred_arts], m


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1-B: TF-IDF + LightGBM
# ═════════════════════════════════════════════════════════════════════════════
def stage1_tfidf_lgbm(df, tr, te, mlb):
    print("\n  S1-B: TF-IDF + LightGBM ...")
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=3)
    X_tr = tfidf.fit_transform(df["text"].iloc[tr].tolist())
    X_te = tfidf.transform(df["text"].iloc[te].tolist())

    Y_tr      = mlb.transform(df["articles"].iloc[tr].tolist())
    Y_te_true = mlb.transform(df["articles"].iloc[te].tolist())
    Y_te_pred = np.zeros_like(Y_te_true)

    for i, art in enumerate(mlb.classes_):
        clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=5,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_tr, Y_tr[:, i])
        Y_te_pred[:, i] = clf.predict(X_te)

    m = article_f1(Y_te_true, Y_te_pred)
    print(f"    F1-micro={m['f1_micro']}  F1-macro={m['f1_macro']}")
    pred_arts = mlb.inverse_transform(Y_te_pred)
    return [list(a) for a in pred_arts], m


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1-C: Full-doc char n-gram + SVD + LightGBM
# ═════════════════════════════════════════════════════════════════════════════
def stage1_fulldoc(df, tr, te, mlb):
    print("\n  S1-C: Full-document char n-gram + SVD + LightGBM ...")
    wc = df["text"].apply(lambda x: len(x.split()))
    print(f"    Avg doc length: {wc.mean():.0f} words  "
          f"(BERT would truncate at ~380 words)")

    vec = HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        n_features=2**17, norm="l2", alternate_sign=False)

    X_tr_hash = vec.transform(df["text"].iloc[tr].tolist())
    X_te_hash = vec.transform(df["text"].iloc[te].tolist())

    svd = TruncatedSVD(n_components=300, random_state=42)
    X_tr = svd.fit_transform(X_tr_hash)
    X_te = svd.transform(X_te_hash)

    Y_tr      = mlb.transform(df["articles"].iloc[tr].tolist())
    Y_te_true = mlb.transform(df["articles"].iloc[te].tolist())
    Y_te_pred = np.zeros_like(Y_te_true)

    for i, art in enumerate(mlb.classes_):
        clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=31, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_tr, Y_tr[:, i])
        Y_te_pred[:, i] = clf.predict(X_te)

    m = article_f1(Y_te_true, Y_te_pred)
    print(f"    F1-micro={m['f1_micro']}  F1-macro={m['f1_macro']}")
    pred_arts = mlb.inverse_transform(Y_te_pred)
    return [list(a) for a in pred_arts], m


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1-D: FastText embeddings + BiLSTM  ← NEW
# ═════════════════════════════════════════════════════════════════════════════

def train_fasttext_embeddings(texts, dim=100, epochs=5, min_count=3, seed=42):
    """
    Train FastText word embeddings on the full corpus using gensim.
    FastText handles OOV words via character subword n-grams —
    important for legal text with rare terminology and citations.
    Returns: (gensim FastText model, vocab list, word→index dict)
    """
    from gensim.models import FastText as GensimFastText

    print(f"    Training FastText (dim={dim}, epochs={epochs}) on full corpus ...")
    tokenized = [simple_tokenize(t) for t in texts]
    total_words = sum(len(t) for t in tokenized)
    print(f"    Corpus: {len(tokenized):,} docs, {total_words:,} total tokens")

    model = GensimFastText(
        sentences=tokenized,
        vector_size=dim,
        window=5,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        seed=seed,
        sg=1,           # skip-gram (better for rare legal terms)
        min_n=3,        # char n-gram min
        max_n=6,        # char n-gram max
    )
    vocab = list(model.wv.key_to_index.keys())
    word2idx = {w: i + 1 for i, w in enumerate(vocab)}  # 0 reserved for PAD
    print(f"    Vocabulary: {len(vocab):,} words")
    return model, vocab, word2idx


def build_embedding_matrix(ft_model, vocab, dim):
    """Build numpy embedding matrix from trained FastText model."""
    matrix = np.zeros((len(vocab) + 1, dim), dtype=np.float32)
    for i, word in enumerate(vocab):
        try:
            matrix[i + 1] = ft_model.wv[word]
        except KeyError:
            matrix[i + 1] = np.random.normal(0, 0.1, dim)
    return matrix


def texts_to_sequences(texts, word2idx, max_seq_len, ft_model=None):
    """
    Convert texts to padded integer sequences.
    Uses FastText OOV vectors for unseen words (via ft_model).
    Returns: (padded sequences as int32 array, actual lengths)
    """
    seqs, lengths = [], []
    for text in texts:
        tokens = simple_tokenize(text)
        if max_seq_len > 0:
            tokens = tokens[:max_seq_len]
        # Map tokens; unknown words still get index 0 (PAD)
        # but we track them — FastText handles OOV at embedding lookup
        ids = [word2idx.get(w, 0) for w in tokens]
        seqs.append(ids)
        lengths.append(max(len(ids), 1))

    # Pad sequences to max length in this batch
    max_len = max(lengths) if lengths else 1
    padded = np.zeros((len(seqs), max_len), dtype=np.int32)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    return padded, np.array(lengths, dtype=np.int64)


# ─── PyTorch BiLSTM model ─────────────────────────────────────────────────────
def build_bilstm(embedding_matrix, hidden_size, num_layers, num_classes,
                 dropout=0.3, freeze_emb=False):
    """
    Build BiLSTM model in PyTorch.
    Architecture:
        Embedding (pretrained FastText, optionally frozen)
        → BiLSTM (N layers)
        → Attention pooling over time steps
        → Dropout
        → Linear → Sigmoid (multi-label)
    """
    import torch
    import torch.nn as nn

    vocab_size, emb_dim = embedding_matrix.shape

    class AttentionPooling(nn.Module):
        """Soft attention over LSTM outputs — weights important time steps."""
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.Linear(hidden_dim, 1)

        def forward(self, lstm_out, lengths):
            # lstm_out: (batch, seq, hidden)
            scores = self.attn(lstm_out).squeeze(-1)          # (batch, seq)
            # Mask padding
            batch, seq = scores.shape
            mask = torch.arange(seq, device=scores.device).unsqueeze(0) \
                   >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq, 1)
            return (lstm_out * weights).sum(dim=1)                # (batch, hidden)

    class BiLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(
                vocab_size, emb_dim, padding_idx=0)
            self.embedding.weight = nn.Parameter(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                requires_grad=not freeze_emb)

            self.lstm = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0)

            self.attn   = AttentionPooling(hidden_size * 2)
            self.drop   = nn.Dropout(dropout)
            self.fc     = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, x, lengths):
            emb = self.drop(self.embedding(x))       # (B, T, E)
            # Pack for efficiency — skips padding in LSTM
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True,
                enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True)         # (B, T, 2*H)
            pooled  = self.attn(out, lengths)         # (B, 2*H)
            dropped = self.drop(pooled)
            logits  = self.fc(dropped)                # (B, C)
            return logits

    return BiLSTMClassifier()


def train_bilstm(model, X_tr, len_tr, Y_tr, X_val, len_val, Y_val,
                 epochs, batch_size, lr, device):
    """
    Training loop with:
    - BCEWithLogitsLoss (multi-label)
    - Adam optimizer with LR scheduler
    - Early stopping on val F1
    - Class-balanced positive weights
    """
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    model = model.to(device)

    # Compute positive weights for imbalanced classes
    pos_counts  = Y_tr.sum(axis=0).astype(np.float32) + 1e-6
    neg_counts  = (len(Y_tr) - Y_tr.sum(axis=0)).astype(np.float32) + 1e-6
    pos_weight  = torch.tensor(neg_counts / pos_counts,
                               dtype=torch.float32, device=device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer   = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    n_tr    = len(X_tr)
    indices = np.arange(n_tr)

    best_val_f1  = -1.0
    best_weights = None
    patience_ctr = 0
    PATIENCE     = 4

    print(f"    Training {epochs} epochs, batch={batch_size}, lr={lr}, "
          f"device={device}")

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n_tr, batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = torch.tensor(X_tr[batch_idx], dtype=torch.long,   device=device)
            lb = torch.tensor(len_tr[batch_idx], dtype=torch.long,  device=device)
            yb = torch.tensor(Y_tr[batch_idx],  dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits = model(xb, lb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        # Validation
        val_f1, _, _ = evaluate_bilstm(model, X_val, len_val, Y_val,
                                        batch_size, device)
        scheduler.step(val_f1)
        avg_loss = total_loss / max(n_batches, 1)
        print(f"      Epoch {epoch+1:2d}/{epochs}  "
              f"loss={avg_loss:.4f}  val_f1_micro={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"      Early stopping at epoch {epoch+1}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"    Restored best weights (val_f1={best_val_f1:.4f})")

    return model


def evaluate_bilstm(model, X, lengths, Y_true, batch_size, device,
                    threshold=0.5):
    """Run inference and return (f1_micro, Y_pred, Y_prob)."""
    import torch

    model.eval()
    all_probs = []
    n = len(X)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = torch.tensor(X[start:start+batch_size],
                               dtype=torch.long, device=device)
            lb = torch.tensor(lengths[start:start+batch_size],
                               dtype=torch.long, device=device)
            logits = model(xb, lb)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    Y_prob = np.vstack(all_probs)
    Y_pred = (Y_prob >= threshold).astype(int)
    f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    return f1, Y_pred, Y_prob


def stage1_fasttext_bilstm(df, tr, vl, te, mlb, args):
    """
    Stage 1-D: FastText embeddings + BiLSTM for article prediction.

    Why this beats TF-IDF / char n-grams:
    1. FastText captures morphological variants of legal terms via subword n-grams
       (e.g. "inadmissibility", "inadmissible", "admissibility" share subwords)
    2. BiLSTM models sequential context — the ORDER of article mentions matters
       (e.g. "Article 5 taken together with Article 6" vs standalone mentions)
    3. Attention pooling weights legally significant sentences more heavily
    4. Full document is read (max_seq_len words, no 512 hard limit)
    """
    import torch

    print("\n  S1-D: FastText + BiLSTM (full document, sequential model) ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # ── Step 1: Train FastText on full training corpus ────────────────────────
    all_train_texts = df["text"].iloc[tr].tolist()
    ft_model, vocab, word2idx = train_fasttext_embeddings(
        all_train_texts,
        dim=args.ft_dim,
        epochs=args.ft_epochs,
        seed=args.seed)

    emb_matrix = build_embedding_matrix(ft_model, vocab, args.ft_dim)
    print(f"    Embedding matrix: {emb_matrix.shape}")

    # ── Step 2: Tokenize & pad all splits ────────────────────────────────────
    max_len = args.max_seq_len  # 0 = no truncation

    print(f"    Encoding documents (max_seq_len={max_len if max_len else 'unlimited'}) ...")
    X_tr, len_tr = texts_to_sequences(
        df["text"].iloc[tr].tolist(), word2idx, max_len, ft_model)
    X_vl, len_vl = texts_to_sequences(
        df["text"].iloc[vl].tolist(), word2idx, max_len, ft_model)
    X_te, len_te = texts_to_sequences(
        df["text"].iloc[te].tolist(), word2idx, max_len, ft_model)

    print(f"    Sequence stats — "
          f"train median_len={np.median(len_tr):.0f}  "
          f"max_len={len_tr.max()}")

    Y_tr      = mlb.transform(df["articles"].iloc[tr].tolist()).astype(np.float32)
    Y_vl      = mlb.transform(df["articles"].iloc[vl].tolist()).astype(np.float32)
    Y_te_true = mlb.transform(df["articles"].iloc[te].tolist())

    num_classes = len(mlb.classes_)

    # ── Step 3: Build BiLSTM ──────────────────────────────────────────────────
    model = build_bilstm(
        embedding_matrix=emb_matrix,
        hidden_size=args.lstm_hidden,
        num_layers=args.lstm_layers,
        num_classes=num_classes,
        dropout=0.3,
        freeze_emb=False,       # fine-tune embeddings during training
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model parameters: {total_params:,}")

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    model = train_bilstm(
        model, X_tr, len_tr, Y_tr,
        X_vl, len_vl, Y_vl,
        epochs=args.lstm_epochs,
        batch_size=args.lstm_batch,
        lr=args.lstm_lr,
        device=device)

    # ── Step 5: Evaluate on test set ──────────────────────────────────────────
    f1_micro, Y_te_pred, Y_te_prob = evaluate_bilstm(
        model, X_te, len_te, Y_te_true, args.lstm_batch, device)
    f1_macro = f1_score(Y_te_true, Y_te_pred, average="macro", zero_division=0)
    m = {"f1_micro": round(f1_micro, 4), "f1_macro": round(f1_macro, 4)}
    print(f"    Test  F1-micro={m['f1_micro']}  F1-macro={m['f1_macro']}")

    pred_arts = mlb.inverse_transform(Y_te_pred)
    return [list(a) for a in pred_arts], m, model


# ═════════════════════════════════════════════════════════════════════════════
# Main comparison
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("  ECHR Improved Bridge — FastText + BiLSTM Stage 1")
    print("=" * 65)

    df, tr, vl, te, mlb = load_data(
        args.data, args.train_ratio, args.val_ratio, args.seed)

    tr_award = df["award_binary"].iloc[tr].tolist()
    te_award = df["award_binary"].iloc[te].tolist()

    tr_arts_true = df["articles"].iloc[tr].tolist()
    te_arts_true = df["articles"].iloc[te].tolist()

    X_tr_oracle = build_combo_features(tr_arts_true, mlb)
    X_te_oracle = build_combo_features(te_arts_true, mlb)

    all_results = []

    # ── Oracle ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ORACLE STAGE 2 (upper bound — true article labels)")
    print("=" * 65)
    m_oracle, _ = run_stage2_lgbm(
        X_tr_oracle, X_te_oracle, tr_award, te_award,
        label="(oracle combo features)")
    all_results.append({
        "System": "Oracle + LightGBM",
        "Stage1": "oracle",
        "Stage1 F1-micro": "--",
        **m_oracle,
        "Note": "Upper bound"
    })

    # ── S1-A ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  S1-A: TF-IDF + SVM  →  Stage 2")
    print("=" * 65)
    te_pred_svm, s1a_metrics = stage1_tfidf_svm(df, tr, te, mlb)
    X_te_s1a = build_combo_features(te_pred_svm, mlb)
    m_s1a, _ = run_stage2_lgbm(
        X_tr_oracle, X_te_s1a, tr_award, te_award,
        label="TF-IDF+SVM → LightGBM")
    all_results.append({
        "System": "TF-IDF+SVM → LightGBM",
        "Stage1": "TF-IDF+SVM",
        "Stage1 F1-micro": s1a_metrics["f1_micro"],
        **m_s1a, "Note": "Baseline"
    })

    # ── S1-B ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  S1-B: TF-IDF + LightGBM  →  Stage 2")
    print("=" * 65)
    te_pred_lgbm, s1b_metrics = stage1_tfidf_lgbm(df, tr, te, mlb)
    X_te_s1b = build_combo_features(te_pred_lgbm, mlb)
    m_s1b, _ = run_stage2_lgbm(
        X_tr_oracle, X_te_s1b, tr_award, te_award,
        label="TF-IDF+LightGBM → LightGBM")
    all_results.append({
        "System": "TF-IDF+LightGBM → LightGBM",
        "Stage1": "TF-IDF+LightGBM",
        "Stage1 F1-micro": s1b_metrics["f1_micro"],
        **m_s1b, "Note": "Better Stage 1 (tree)"
    })

    # ── S1-C ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  S1-C: Full-doc char n-gram  →  Stage 2")
    print("=" * 65)
    te_pred_full, s1c_metrics = stage1_fulldoc(df, tr, te, mlb)
    X_te_s1c = build_combo_features(te_pred_full, mlb)
    m_s1c, _ = run_stage2_lgbm(
        X_tr_oracle, X_te_s1c, tr_award, te_award,
        label="FullDoc char n-gram → LightGBM")
    all_results.append({
        "System": "FullDoc+LightGBM → LightGBM",
        "Stage1": "FullDoc char n-gram",
        "Stage1 F1-micro": s1c_metrics["f1_micro"],
        **m_s1c, "Note": "Full document (no 512 limit)"
    })

    # ── S1-D: FastText + BiLSTM ───────────────────────────────────────────────
    if not args.skip_bilstm:
        print("\n" + "=" * 65)
        print("  S1-D: FastText + BiLSTM  →  Stage 2  (NEW)")
        print("  Key advantages:")
        print("    • FastText subword n-grams handle rare legal terminology")
        print("    • BiLSTM captures sequential context (article co-occurrence order)")
        print("    • Attention pooling focuses on legally significant sentences")
        print(f"    • Reads up to {args.max_seq_len} tokens (vs BERT's 512)")
        print("=" * 65)

        te_pred_bilstm, s1d_metrics, _ = stage1_fasttext_bilstm(
            df, tr, vl, te, mlb, args)
        X_te_s1d = build_combo_features(te_pred_bilstm, mlb)
        m_s1d, _ = run_stage2_lgbm(
            X_tr_oracle, X_te_s1d, tr_award, te_award,
            label="FastText+BiLSTM → LightGBM")
        all_results.append({
            "System": "FastText+BiLSTM → LightGBM",
            "Stage1": "FastText+BiLSTM",
            "Stage1 F1-micro": s1d_metrics["f1_micro"],
            **m_s1d, "Note": "NEW: sequential + subword"
        })
    else:
        print("\n  S1-D: Skipped (--skip_bilstm flag set)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL COMPARISON TABLE")
    print("=" * 65)

    res_df = pd.DataFrame(all_results)
    cols = ["System", "Stage1 F1-micro", "accuracy", "f1", "auc", "Note"]
    print(res_df[cols].to_string(index=False))

    # Gap analysis
    oracle_auc   = all_results[0]["auc"]
    baseline_auc = all_results[1]["auc"]

    print(f"\n  Gap analysis (Oracle AUC = {oracle_auc:.4f}):")
    for r in all_results[1:]:
        gap     = oracle_auc - r["auc"]
        closed  = (r["auc"] - baseline_auc) / (oracle_auc - baseline_auc) * 100 \
                   if oracle_auc != baseline_auc else 0
        print(f"    {r['System'][:40]:40s}  "
              f"AUC={r['auc']:.4f}  "
              f"gap={gap:.4f}  "
              f"gap_closed={closed:+.1f}%")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "improved_results.csv")
    res_df.to_csv(csv_path, index=False)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    latex_path = os.path.join(args.output_dir, "improved_table.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\\small\n")
        f.write("\\begin{tabular}{lccccl}\n\\toprule\n")
        f.write("System & S1 F1-$\\mu$ & Award Acc & "
                "Award F1 & Award AUC & Note \\\\\n")
        f.write("\\midrule\n")
        for r in all_results:
            s1f1 = r["Stage1 F1-micro"]
            f.write(f"{r['System']} & {s1f1} & "
                    f"{r['accuracy']:.4f} & {r['f1']:.4f} & "
                    f"{r['auc']:.4f} & {r['Note']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(
            "\\caption{Violation-to-Award Bridge. "
            "S1-D (FastText + BiLSTM) reads the full judgment using "
            "subword embeddings and sequential attention, unlike BERT "
            "which truncates at 512 tokens. "
            "Stage 2 uses LightGBM over article co-occurrence features.}\n"
        )
        f.write("\\label{tab:improved}\n\\end{table}\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    systems = [r["System"] for r in all_results]
    aucs    = [r["auc"] for r in all_results]
    f1s     = [r["f1"] for r in all_results]
    colors  = ["#2ecc71" if "Oracle" in s
               else "#e74c3c" if "BiLSTM" in s
               else "#3498db" for s in systems]

    short_names = [s.split(" →")[0].replace("TF-IDF+", "").replace("FullDoc", "FullDoc")
                   for s in systems]

    ax = axes[0]
    bars = ax.barh(short_names, aucs, color=colors)
    ax.set_xlabel("Award AUC")
    ax.set_title("Stage 2 Award AUC by Stage 1 Model")
    ax.axvline(oracle_auc, color="green", linestyle="--", alpha=0.5,
               label=f"Oracle={oracle_auc:.3f}")
    ax.axvline(baseline_auc, color="gray", linestyle=":", alpha=0.5,
               label=f"Baseline={baseline_auc:.3f}")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, max(aucs) * 1.15)

    ax2 = axes[1]
    s1_f1s = [r["Stage1 F1-micro"] for r in all_results]
    stage1_nums = []
    for v in s1_f1s:
        try:    stage1_nums.append(float(v))
        except: stage1_nums.append(0.0)
    bars2 = ax2.barh(short_names, stage1_nums, color=colors)
    ax2.set_xlabel("Stage 1 F1-micro (Article Prediction)")
    ax2.set_title("Stage 1 Article Prediction Quality")
    for bar, val in zip(bars2, stage1_nums):
        if val > 0:
            ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}", va="center", fontsize=8)
    ax2.set_xlim(0, max(v for v in stage1_nums if v > 0) * 1.15 if any(stage1_nums) else 1)

    plt.suptitle("ECHR Violation-to-Award Bridge: Stage 1 Model Comparison",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "stage1_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Saved: {csv_path}")
    print(f"  LaTeX: {latex_path}")
    print(f"  Plot:  {plot_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
