#!/usr/bin/env python3
"""
Unmix the Signal: Blind Source Separation with FastICA implemented from scratch.

Pipeline
--------
1. Load N clean sources (WAV files from data/sources/, or synthesize demo sources).
2. Mix them into M >= N observed signals with our own mixing matrix A:  X = A @ S
3. Center the mixtures.
4. PCA-whiten using covariance matrix + eigendecomposition (np.linalg.eigh only).
5. FastICA (symmetric or deflation) implemented from scratch. No library ICA.
6. Resolve scale/sign/permutation ambiguity, export separated WAVs.
7. Evaluate: |correlation|, SDR, SI-SDR.  Bonus: PCA-vs-ICA and SNR sweep.

Only numpy / scipy / matplotlib are used.
scipy is used for WAV I/O, resampling and the Hungarian matching,
never for ICA, PCA or whitening.

Usage
-----
    python ica_audio.py                       # synthetic demo sources
    python ica_audio.py --sources data/sources   # your own clean WAVs
    python ica_audio.py --algo deflation --g logcosh --snr-sweep
"""

import argparse
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; plots are saved to disk
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly

EPS = 1e-12


# =============================================================================
# 1. DATA: loading / synthesizing clean sources
# =============================================================================
def load_wav_mono(path, target_sr, max_seconds):
    """Read a WAV, convert to mono float64 in [-1, 1], resample, truncate."""
    sr, raw = wavfile.read(path)
    x = raw.astype(np.float64)
    if x.ndim > 1:                       # stereo -> mono
        x = x.mean(axis=1)
    if np.issubdtype(raw.dtype, np.integer):   # int PCM -> [-1, 1]
        x /= np.iinfo(raw.dtype).max
    if sr != target_sr:                  # polyphase resample
        g = np.gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g)
    return x[: int(max_seconds * target_sr)]


def load_sources(folder, sr=16000, seconds=10.0):
    """Load every .wav in `folder` as a source; trim all to equal length."""
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    if len(files) < 2:
        raise FileNotFoundError(f"Need >= 2 WAV files in {folder}")
    sigs = [load_wav_mono(f, sr, seconds) for f in files]
    n = min(len(s) for s in sigs)
    S = np.stack([s[:n] for s in sigs])
    S /= np.max(np.abs(S), axis=1, keepdims=True) + EPS   # peak-normalise
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return S, names


def synth_sources(sr=16000, seconds=10.0, seed=0):
    """
    Three synthetic, clearly NON-Gaussian sources (a speech-like, a music-like
    and an ambient-like signal) so the demo runs with zero external files.
    Replace with real recordings for the final submission.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr

    # (a) "speech": harmonic buzz with wandering pitch gated by syllable bursts
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t) + 10 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voiced = sum(np.sin(k * phase) / k for k in range(1, 12))
    syll = np.clip(np.sin(2 * np.pi * 3.0 * t) + 0.3 * np.sin(2 * np.pi * 0.9 * t), 0, None) ** 2
    speech = voiced * syll

    # (b) "music": plucked notes (decaying harmonics) in a repeating melody
    notes = [262, 330, 392, 330, 294, 349, 440, 349]      # Hz
    step = int(0.35 * sr)
    music = np.zeros(n)
    for i in range(0, n, step):
        fr = notes[(i // step) % len(notes)]
        seg_t = np.arange(min(step * 2, n - i)) / sr
        env = np.exp(-6 * seg_t)
        tone = sum(np.sin(2 * np.pi * fr * h * seg_t) / h**1.5 for h in range(1, 5))
        music[i:i + len(seg_t)] += env * tone

    # (c) "ambient": sparse crackles / rain drops (impulsive, super-Gaussian)
    imp = (rng.random(n) < 0.002) * rng.laplace(size=n)
    kern = np.exp(-np.arange(400) / 60.0)
    ambient = np.convolve(imp, kern, mode="same")

    S = np.stack([speech, music, ambient])
    S -= S.mean(axis=1, keepdims=True)
    S /= np.max(np.abs(S), axis=1, keepdims=True) + EPS
    return S, ["speech_like", "music_like", "ambient_like"]


# =============================================================================
# 2. MIXING
# =============================================================================
def make_mixing_matrix(n_mix, n_src, seed=1, min_cond=1.5, max_cond=30.0):
    """
    Random mixing matrix with entries in [0.2, 1.0]; re-draw until
    well-conditioned, since ICA needs A to be (numerically) invertible.
    """
    rng = np.random.default_rng(seed)
    while True:
        A = rng.uniform(0.2, 1.0, size=(n_mix, n_src))
        c = np.linalg.cond(A)
        if min_cond < c < max_cond:
            return A


def mix(S, A, snr_db=None, seed=2):
    """X = A S (+ optional white sensor noise at the requested SNR in dB)."""
    X = A @ S
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p_sig = np.mean(X**2, axis=1, keepdims=True)
        p_noise = p_sig / (10 ** (snr_db / 10))
        X = X + rng.normal(size=X.shape) * np.sqrt(p_noise)
    return X


# =============================================================================
# 3. PREPROCESSING: centering + PCA whitening
# =============================================================================
def center(X):
    """Subtract the per-channel mean. Returns (X_centered, mean)."""
    mu = X.mean(axis=1, keepdims=True)
    return X - mu, mu


def pca_whiten(Xc, n_components=None):
    """
    PCA whitening from scratch.

        C = (1/T) Xc Xc^T              covariance matrix  (M x M)
        C = E D E^T                    eigendecomposition (E orthogonal, D diagonal)
        V = D^{-1/2} E^T               whitening matrix
        Z = V Xc                       => cov(Z) = I

    If n_components < M we keep only the largest eigenvalues, which also
    reduces dimension for over-determined mixtures (M > N).

    Returns Z (whitened), V (whitening), Vinv (de-whitening), eigvals, eigvecs.
    """
    T = Xc.shape[1]
    C = (Xc @ Xc.T) / T
    eigvals, eigvecs = np.linalg.eigh(C)        # ascending order, symmetric C
    order = np.argsort(eigvals)[::-1]           # sort descending
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    k = n_components or len(eigvals)
    D = eigvals[:k]
    E = eigvecs[:, :k]
    V = np.diag(1.0 / np.sqrt(D + EPS)) @ E.T   # whitening matrix
    Vinv = E @ np.diag(np.sqrt(D + EPS))        # its pseudo-inverse
    Z = V @ Xc
    return Z, V, Vinv, eigvals, eigvecs


# =============================================================================
# 4. FastICA FROM SCRATCH
# =============================================================================
# Contrast-function derivatives. Each returns (g(u), g'(u)) with u = w^T z.
def _g_logcosh(u, a=1.0):
    t = np.tanh(a * u)
    return t, a * (1.0 - t**2)

def _g_exp(u):
    e = np.exp(-(u**2) / 2.0)
    return u * e, (1.0 - u**2) * e

def _g_cube(u):                          # kurtosis-based
    return u**3, 3.0 * u**2

NONLINEARITIES = {"logcosh": _g_logcosh, "exp": _g_exp, "cube": _g_cube}


def _sym_decorrelate(W):
    """W <- (W W^T)^{-1/2} W   (symmetric orthogonalisation via eigh)."""
    s, U = np.linalg.eigh(W @ W.T)
    return U @ np.diag(1.0 / np.sqrt(s + EPS)) @ U.T @ W


def fastica(Z, n_components=None, algo="symmetric", g="logcosh",
            max_iter=500, tol=1e-7, seed=0, verbose=True):
    """
    FastICA on whitened data Z (shape: n_dims x T), cov(Z) = I.

    Fixed-point update for one unit (Hyvarinen 1999):
        w+ = E[ z g(w^T z) ] - E[ g'(w^T z) ] w
        w  = w+ / ||w||
    Because Z is white, the demixing matrix W is orthogonal, so we keep the
    rows decorrelated either by
        * 'symmetric' : update all rows in parallel, then orthogonalise W
        * 'deflation' : extract one row at a time, Gram-Schmidt against the
                        previously found rows.
    Returns W (k x k) such that  Y = W Z  are the independent components.
    """
    rng = np.random.default_rng(seed)
    k = n_components or Z.shape[0]
    T = Z.shape[1]
    gfun = NONLINEARITIES[g]

    if algo == "symmetric":
        W = _sym_decorrelate(rng.normal(size=(k, k)))
        for it in range(max_iter):
            U = W @ Z                                   # (k, T)
            gu, gpu = gfun(U)
            W_new = (gu @ Z.T) / T - np.diag(gpu.mean(axis=1)) @ W
            W_new = _sym_decorrelate(W_new)
            # convergence: each new row parallel to the old one (|<w_new, w_old>| -> 1)
            delta = np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0))
            W = W_new
            if delta < tol:
                break
        if verbose:
            print(f"  FastICA (symmetric, g={g}) converged in {it + 1} iterations")
        return W

    elif algo == "deflation":
        W = np.zeros((k, k))
        for p in range(k):
            w = rng.normal(size=k)
            w /= np.linalg.norm(w)
            for it in range(max_iter):
                u = w @ Z
                gu, gpu = gfun(u)
                w_new = (Z * gu).mean(axis=1) - gpu.mean() * w
                # Gram-Schmidt: remove projections on already-found components
                w_new -= W[:p].T @ (W[:p] @ w_new)
                w_new /= np.linalg.norm(w_new)
                done = abs(abs(w_new @ w) - 1.0) < tol
                w = w_new
                if done:
                    break
            W[p] = w
            if verbose:
                print(f"  FastICA (deflation) component {p + 1}/{k}: {it + 1} iterations")
        return W
    raise ValueError("algo must be 'symmetric' or 'deflation'")


# =============================================================================
# 5. EVALUATION (handles scale, sign and permutation ambiguity)
# =============================================================================
def corr_matrix(Y, S):
    """|Pearson correlation| between every estimate (rows of Y) and source (rows of S)."""
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Sc = S - S.mean(axis=1, keepdims=True)
    Yn = Yc / (np.linalg.norm(Yc, axis=1, keepdims=True) + EPS)
    Sn = Sc / (np.linalg.norm(Sc, axis=1, keepdims=True) + EPS)
    return np.abs(Yn @ Sn.T)


def match_and_rescale(Y, S):
    """
    ICA recovers sources only up to permutation, sign and scale.
      * Permutation: Hungarian assignment maximising |corr|.
      * Sign + scale: least-squares gain  alpha = <y, s>/<y, y>  per source
        (the sign of alpha fixes polarity, its magnitude fixes scale).
    Returns Y_aligned (same order & scale as S) and the permutation.
    """
    C = corr_matrix(Y, S)
    rows, cols = linear_sum_assignment(-C)          # maximise total correlation
    perm = np.zeros(S.shape[0], dtype=int)
    perm[cols] = rows                               # perm[j] = estimate index for source j
    Ya = np.zeros_like(S)
    for j in range(S.shape[0]):
        y = Y[perm[j]] - Y[perm[j]].mean()
        s = S[j] - S[j].mean()
        alpha = (y @ s) / (y @ y + EPS)
        Ya[j] = alpha * y
    return Ya, perm


def sdr_db(est, ref):
    """
    Signal-to-Distortion Ratio (dB) after optimal scaling of the estimate:
    project `ref` direction out of `est`; anything left is distortion.
        SDR = 10 log10( ||s_target||^2 / ||e||^2 ),
        s_target = <est, ref>/<ref, ref> * ref   (= SI-SDR, scale invariant)
    """
    est = est - est.mean()
    ref = ref - ref.mean()
    s_target = (est @ ref) / (ref @ ref + EPS) * ref
    e = est - s_target
    return 10 * np.log10((s_target @ s_target + EPS) / (e @ e + EPS))


def evaluate(Y_aligned, S):
    """Per-source |corr| and SI-SDR for already-aligned estimates."""
    corr = np.array([abs(np.corrcoef(Y_aligned[j], S[j])[0, 1]) for j in range(len(S))])
    sdr = np.array([sdr_db(Y_aligned[j], S[j]) for j in range(len(S))])
    return corr, sdr


def excess_kurtosis(X):
    """Excess kurtosis per row (0 for a Gaussian)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    return (Xc**4).mean(axis=1) / (Xc**2).mean(axis=1) ** 2 - 3.0


# =============================================================================
# 6. EXPORT
# =============================================================================
def save_wav(path, x, sr):
    """Peak-normalise to 0.9 and write as 16-bit PCM."""
    x = x - np.mean(x)
    x = 0.9 * x / (np.max(np.abs(x)) + EPS)
    wavfile.write(path, sr, (x * 32767).astype(np.int16))


# =============================================================================
# 7. PLOTTING
# =============================================================================
def plot_waveforms(rows, titles, sr, path, suptitle, max_s=3.0):
    n = len(rows)
    fig, ax = plt.subplots(n, 1, figsize=(10, 1.6 * n), sharex=True)
    ax = np.atleast_1d(ax)
    m = int(max_s * sr)
    t = np.arange(m) / sr
    for a, r, ti in zip(ax, rows, titles):
        a.plot(t, r[:m], lw=0.5)
        a.set_ylabel(ti, rotation=0, ha="right", va="center", fontsize=8)
        a.set_yticks([])
    ax[-1].set_xlabel("time (s)")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter_stages(S, Xc, Z, Y, path):
    """Joint scatter of first two channels at each stage: the geometric story of ICA."""
    stages = [("Sources S", S), ("Mixed X (centered)", Xc),
              ("PCA-whitened Z", Z), ("ICA output Y", Y)]
    idx = np.random.default_rng(0).choice(S.shape[1], 4000, replace=False)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    for a, (name, M) in zip(ax, stages):
        a.scatter(M[0, idx], M[1, idx], s=1, alpha=0.4)
        a.set_title(name, fontsize=10)
        a.set_aspect("equal", "datalim")
    fig.suptitle("Geometry of ICA: mix, then whiten (decorrelate + unit variance), then rotate (ICA)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_corr_heatmap(C_ica, C_pca, names, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, C, ti in zip(ax, [C_pca, C_ica],
                        ["PCA components vs sources", "ICA outputs vs sources"]):
        im = a.imshow(C, vmin=0, vmax=1, cmap="viridis")
        a.set_title(ti, fontsize=10)
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=30, fontsize=8)
        a.set_yticks(range(C.shape[0]))
        a.set_yticklabels([f"est {i}" for i in range(C.shape[0])], fontsize=8)
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                a.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                       color="w" if C[i, j] < 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(S, X, Y, names, path):
    """Marginals: sources/outputs are peaky (non-Gaussian); mixtures look more Gaussian (CLT)."""
    fig, ax = plt.subplots(3, S.shape[0], figsize=(3.2 * S.shape[0], 6.5))
    for row, (M, lab) in enumerate([(S, "source"), (X, "mixture"), (Y, "ICA output")]):
        for j in range(S.shape[0]):
            m = M[j] / (M[j].std() + EPS)
            ax[row, j].hist(m, bins=100, range=(-5, 5), density=True, alpha=0.8)
            xs = np.linspace(-5, 5, 200)
            ax[row, j].plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "r--", lw=0.8)
            ax[row, j].set_title(f"{lab} {j}  kurt={excess_kurtosis(m[None])[0]:.1f}",
                                 fontsize=8)
            ax[row, j].set_yscale("log")
            ax[row, j].set_ylim(1e-4, 5)
    fig.suptitle("Amplitude histograms (red dashed = Gaussian)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_snr_sweep(snrs, ica_sdr, pca_sdr, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(snrs, ica_sdr, "o-", label="ICA (FastICA)")
    ax.plot(snrs, pca_sdr, "s--", label="PCA only")
    ax.set_xlabel("Sensor SNR (dB)")
    ax.set_ylabel("Mean SI-SDR of separated sources (dB)")
    ax.set_title("Separation quality vs. noise level")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# 8. FULL PIPELINE (reused by the SNR sweep)
# =============================================================================
def run_pipeline(S, A, snr_db=None, algo="symmetric", g="logcosh", seed=0, verbose=True):
    N = S.shape[0]
    X = mix(S, A, snr_db=snr_db)
    Xc, mu = center(X)
    Z, V, Vinv, eigvals, _ = pca_whiten(Xc, n_components=N)
    W = fastica(Z, n_components=N, algo=algo, g=g, seed=seed, verbose=verbose)
    Y = W @ Z
    Y_al, perm = match_and_rescale(Y, S)
    Y_pca_al, perm_pca = match_and_rescale(Z, S)       # PCA baseline: whitened PCs only
    return dict(X=X, Xc=Xc, Z=Z, V=V, W=W, Y=Y, Y_al=Y_al, perm=perm,
                Zpca_al=Y_pca_al, eigvals=eigvals)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="FastICA audio unmixing from scratch")
    ap.add_argument("--sources", default=None, help="folder with clean .wav sources")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--n-mix", type=int, default=None, help="#mixtures (>= #sources)")
    ap.add_argument("--algo", choices=["symmetric", "deflation"], default="symmetric")
    ap.add_argument("--g", choices=list(NONLINEARITIES), default="logcosh")
    ap.add_argument("--snr", type=float, default=None, help="sensor noise SNR in dB")
    ap.add_argument("--snr-sweep", action="store_true", help="run bonus SNR experiment")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "audio"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)

    # ---- 1. sources --------------------------------------------------------
    if args.sources:
        S, names = load_sources(args.sources, args.sr, args.seconds)
    else:
        print("No --sources given: using synthetic demo sources.")
        S, names = synth_sources(args.sr, args.seconds)
    N = S.shape[0]
    M = args.n_mix or N
    print(f"{N} sources, {M} mixtures, {S.shape[1]} samples @ {args.sr} Hz")

    # ---- 2-6. mix -> center -> whiten -> ICA -> align ----------------------
    A = make_mixing_matrix(M, N, seed=args.seed)
    print("Mixing matrix A =\n", np.round(A, 3), f"\n cond(A) = {np.linalg.cond(A):.2f}")
    r = run_pipeline(S, A, snr_db=args.snr, algo=args.algo, g=args.g)

    # sanity checks that the maths did what the theory says
    cov_Z = (r["Z"] @ r["Z"].T) / r["Z"].shape[1]
    print(f"Whitening check  max|cov(Z) - I| = {np.abs(cov_Z - np.eye(N)).max():.2e}")
    print(f"Demixing orthogonality  max|W W^T - I| = {np.abs(r['W'] @ r['W'].T - np.eye(N)).max():.2e}")

    # ---- 7. metrics --------------------------------------------------------
    corr_ica, sdr_ica = evaluate(r["Y_al"], S)
    corr_pca, sdr_pca = evaluate(r["Zpca_al"], S)
    corr_mix = np.array([max(abs(np.corrcoef(r["X"][i], S[j])[0, 1]) for i in range(M))
                         for j in range(N)])

    print("\n=== RESULTS ===")
    print(f"{'source':<16}{'|corr| mix':>11}{'|corr| PCA':>11}{'|corr| ICA':>11}"
          f"{'SDR PCA':>10}{'SDR ICA':>10}")
    for j in range(N):
        print(f"{names[j]:<16}{corr_mix[j]:>11.3f}{corr_pca[j]:>11.3f}{corr_ica[j]:>11.3f}"
              f"{sdr_pca[j]:>10.2f}{sdr_ica[j]:>10.2f}")
    print(f"{'MEAN':<16}{corr_mix.mean():>11.3f}{corr_pca.mean():>11.3f}{corr_ica.mean():>11.3f}"
          f"{sdr_pca.mean():>10.2f}{sdr_ica.mean():>10.2f}")
    print("\nExcess kurtosis  sources :", np.round(excess_kurtosis(S), 2))
    print("                 mixtures:", np.round(excess_kurtosis(r["X"]), 2))
    print("                 ICA out :", np.round(excess_kurtosis(r["Y_al"]), 2))

    # ---- 8. export audio ---------------------------------------------------
    for j in range(N):
        save_wav(os.path.join(args.out, "audio", f"source_{j}_{names[j]}.wav"), S[j], args.sr)
        save_wav(os.path.join(args.out, "audio", f"mixture_{j}.wav"), r["X"][j % M], args.sr)
        save_wav(os.path.join(args.out, "audio", f"separated_{j}_{names[j]}.wav"),
                 r["Y_al"][j], args.sr)
        save_wav(os.path.join(args.out, "audio", f"pca_only_{j}_{names[j]}.wav"),
                 r["Zpca_al"][j], args.sr)
    print(f"\nWAV files written to {args.out}/audio/")

    # ---- 9. plots ----------------------------------------------------------
    P = lambda f: os.path.join(args.out, "plots", f)
    plot_waveforms(S, names, args.sr, P("01_sources.png"), "Original clean sources")
    plot_waveforms(r["X"], [f"mix {i}" for i in range(M)], args.sr,
                   P("02_mixtures.png"), "Observed mixtures")
    plot_waveforms(r["Y_al"], [f"ICA {n}" for n in names], args.sr,
                   P("03_separated.png"), "ICA-separated signals (aligned)")
    plot_scatter_stages(S, r["Xc"], r["Z"], r["Y"], P("04_geometry.png"))
    C_ica = corr_matrix(r["Y"], S)
    C_pca = corr_matrix(r["Z"], S)
    plot_corr_heatmap(C_ica, C_pca, names, P("05_correlation_matrix.png"))
    plot_histograms(S, r["X"], r["Y_al"], names, P("06_histograms.png"))

    # ---- 10. bonus: SNR sweep ---------------------------------------------
    if args.snr_sweep:
        print("\nSNR sweep (mean SI-SDR in dB):")
        snrs = [0, 5, 10, 15, 20, 30, 40]
        ica_s, pca_s = [], []
        for snr in snrs:
            rr = run_pipeline(S, A, snr_db=snr, algo=args.algo, g=args.g, verbose=False)
            ica_s.append(evaluate(rr["Y_al"], S)[1].mean())
            pca_s.append(evaluate(rr["Zpca_al"], S)[1].mean())
            print(f"  SNR {snr:>2} dB:  ICA {ica_s[-1]:6.2f}   PCA {pca_s[-1]:6.2f}")
        plot_snr_sweep(snrs, ica_s, pca_s, P("07_snr_sweep.png"))

    print("Plots written to", os.path.join(args.out, "plots"))


if __name__ == "__main__":
    main()
