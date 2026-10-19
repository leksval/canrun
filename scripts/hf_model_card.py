"""Hugging Face model card (README.md) for canrun-fps-predictor.

The Hub renders only $$ block math, so inline $...$ is rewritten to Unicode
text by inline_math_to_text before the card is written.
"""

from __future__ import annotations

import re
from typing import Any

MODEL_NAME = "canrun-fps-predictor"
DEFAULT_REPO_ID = "grasant/canrun-fps-predictor"

# Independent test results (scripts/ds_pipeline/evaluate_*.py), model alone.
EXTERNAL_MODEL_RESULTS = [
    ("TechPowerUp RTX 5070 review, 7 games outside the CanRun table", "techpowerup-rtx5070-2025", 266, 18.9),
    ("TechSpot reviews 2025-2026", "techspot-2025-2026", 98, 36.7),
    ("OpenML 42737 fps-in-video-games", "openml-42737", 12032, 39.4),
]

_LATEX_SYMBOLS = {
    r"\times": "×", r"\rightarrow": "→", r"\ge": "≥", r"\le": "≤", r"\cdot": "·",
    r"\in": "∈", r"\left|": "|", r"\right|": "|", r"\left(": "(", r"\right)": ")",
    r"\log": "log", r"\max": "max", r"\;": " ", r"\_": "_", r"\{": "{", r"\}": "}",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\mu": "μ", r"\sigma": "σ",
}
_SUPERSCRIPTS = str.maketrans("0123456789()-k", "⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾⁻ᵏ")
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)")


def _latex_to_text(expr: str) -> str:
    text = expr
    for _ in range(3):
        text = re.sub(r"\\hat\{(\w)\}", lambda m: m.group(1) + "\u0302", text)
        text = re.sub(r"\\mathbb\{R\}", "ℝ", text)
        text = re.sub(r"\\mathbb\{I\}", "𝟙", text)
        text = re.sub(r"\\(?:text|mathbf|mathrm)\{([^{}]*)\}", r"\1", text)
    # Longest first, so \left| is not read as \le.
    for latex in sorted(_LATEX_SYMBOLS, key=len, reverse=True):
        text = text.replace(latex, _LATEX_SYMBOLS[latex])
    text = re.sub(r"\^\{([0-9()\-k]+)\}|\^([0-9])", lambda m: (m.group(1) or m.group(2)).translate(_SUPERSCRIPTS), text)
    text = re.sub(r"_\{([0-9]+)\}|_([0-9])\b", lambda m: (m.group(1) or m.group(2)).translate(_SUBSCRIPTS), text)
    text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)
    return re.sub(r"\s+", " ", text).strip()


def inline_math_to_text(markdown: str) -> str:
    """Rewrite inline $...$ math as Unicode text; the Hub renders only $$ blocks.

    Results with an underscore go in a code span so the renderer cannot read
    the underscores as italics.
    """

    def repl(match: re.Match[str]) -> str:
        text = _latex_to_text(match.group(1))
        return f"`{text}`" if "_" in text else text

    return _INLINE_MATH.sub(repl, markdown)


def _model_index(summary: dict[str, Any], m4_games: dict[str, Any], m4_gpus: dict[str, Any]) -> str:
    lines = ["model-index:", f"  - name: {MODEL_NAME}", "    results:"]
    for name, dataset_type, points, mdape in EXTERNAL_MODEL_RESULTS:
        lines += [
            "      - task:",
            "          type: tabular-regression",
            "          name: FPS prediction on independent benchmarks (model alone)",
            "        dataset:",
            f"          name: {name} ({points:,} points)",
            f"          type: {dataset_type}",
            "        metrics:",
            "          - type: mdape",
            f"            value: {mdape}",
            "            name: Median absolute percentage error (%)",
        ]
    for label, metrics in (("unseen games", m4_games), ("unseen GPUs", m4_gpus)):
        lines += [
            "      - task:",
            "          type: tabular-regression",
            f"          name: Internal cross-validation, {label} (mostly generated targets)",
            "        dataset:",
            f"          name: CanRun synthetic benchmark suite ({summary['total_records']:,} rows)",
            "          type: canrun-synthetic-suite-v1",
            "        metrics:",
            "          - type: mdape",
            f"            value: {metrics['MdAPE_pct']}",
            "            name: Median absolute percentage error (%)",
            "          - type: r2",
            f"            value: {metrics['R2']}",
            "            name: Raw-FPS R2",
        ]
    return "\n".join(lines)


def _cv_rows(cv: dict[str, Any]) -> str:
    labels = [
        ("M0_Legacy_Naive_Spec_Heuristic", "Spec heuristic, 60 × (user/spec)^0.85"),
        ("M1_Legacy_v8_Lookup_Model", "Legacy v8.0 lookup model"),
        ("M2_Physics_Log_Ridge_Prior_Only", "Stage 0 ridge prior only"),
        ("M3_Single_Stage_Unconstrained_GBDT", "Single unconstrained GBDT"),
        ("M4_New_v9_TwoStage_LogRidge_Monotonic_GBDT", "**This model**"),
    ]
    rows = []
    for key, label in labels:
        m = cv[key]
        rows.append(
            f"| {label} | {m['MdAPE_pct']:.2f}% | {m['MAPE_pct']:.2f}% | {m['R2']:.4f} "
            f"| {m['MAE_fps']:.2f} | {m['Monotonicity_Violation_pct']:.2f}% |"
        )
    return "\n".join(rows)


_ARCHITECTURE_MD = r"""## Architecture

The model predicts log FPS in two stages. A linear prior captures the power law between hardware and frame rate, and a small tree ensemble corrects what the prior misses.

```mermaid
flowchart LR
    Inputs["Hardware + Steam store specs"] -->|"21 features"| X["x"]
    X -->|"standardize + ridge"| S0["Stage 0: log-FPS prior"]
    X -->|"100 monotonic trees"| S1["Stage 1: log residual"]
    S0 --> Sum["clip(prior + residual)"]
    S1 --> Sum
    Sum -->|"exp, 60 FPS lock rule"| FPS["Native raster FPS"]
```

**Stage 0, ridge prior.** Frame rate scales roughly as a power of GPU throughput and inversely with pixel count:

$$\text{FPS} \propto \left(\frac{\text{GPU}_{\text{PassMark}}}{\text{RecGPU}_{\text{PassMark}}}\right)^{\beta} \cdot \left(\frac{\text{Pixels}}{\text{Pixels}_{1080\text{p}}}\right)^{-\gamma}$$

In log space this is linear, so Stage 0 is a ridge regression ($\alpha = 10$) on the 21 standardized features:

$$\hat{y}_{\text{prior}}(\mathbf{x}) = b + \sum_{j=1}^{21} w_j \frac{x_j - \mu_j}{\sigma_j}$$

The linear prior lets the model extrapolate to GPUs faster than any in training, where trees alone would flatten out.

**Stage 1, monotonic residual trees.** 100 gradient-boosted trees (depth 5, at least 10 samples per leaf, learning rate 0.055, L2 1.2, Huber loss) fit the residual $r_i = \log(\text{FPS}_i) - \hat{y}_{\text{prior}}(\mathbf{x}_i)$. Each tree respects the monotonic constraints in the feature table, so a faster GPU or CPU, more VRAM or a lower resolution never lowers the prediction.

$$\hat{y}_{\log}(\mathbf{x}) = \text{clip}\left(\hat{y}_{\text{prior}}(\mathbf{x}) + r_0 + \sum_{m=1}^{100} T_m(\mathbf{x}),\; \log 5,\; \log 500\right)$$

$$\widehat{\text{FPS}}(\mathbf{x}) = \exp\left(\hat{y}_{\log}(\mathbf{x})\right)$$

**Prediction interval.** `predict_with_interval` adds a split-conformal band from out-of-fold log residuals in the synthetic suite. It holds the true value 80% of the time inside that suite, but only 15.4% of the time on TechPowerUp, so do not read it as a real-world 80% interval. A ±30% band is closer to real behavior.

"""

_FEATURES_MD = r"""## Features

No game names, app IDs or GPU names reach the model. All 21 features are computed in `inference.py` (`extract_feature_vector`). A constraint of +1 means the prediction can only rise with the feature, −1 that it can only fall, 0 unconstrained.

| # | Feature | Meaning | Constraint |
| :-: | :--- | :--- | :-: |
| 0 | `log_gpu_passmark` | log of the GPU's PassMark G3D score | +1 |
| 1 | `log_cpu_passmark` | log of the CPU's PassMark score | +1 |
| 2 | `gpu_vram_gb` | GPU memory in GB | +1 |
| 3 | `gpu_arch_code` | Architecture generation, 1.0 (Pascal) to 5.0 (Blackwell) | +1 |
| 4 | `cpu_to_gpu_log_ratio` | 0.75 log GPU + 0.25 log CPU | +1 |
| 5 | `gpu_tier_score` | GPU PassMark / 40,000 | +1 |
| 6 | `log_pixel_ratio` | log(pixels / 1080p pixels): 0 at 1080p, 0.58 at 1440p, 1.39 at 4K | −1 |
| 7 | `vram_per_mpixel` | VRAM per display megapixel | +1 |
| 8 | `is_4k` | 1 at 4K | −1 |
| 9 | `log_rec_gpu_passmark` | log PassMark of the store's recommended GPU | 0 |
| 10 | `log_min_gpu_passmark` | log PassMark of the store's minimum GPU | 0 |
| 11 | `gpu_headroom_rec` | log GPU − log recommended GPU | 0 |
| 12 | `gpu_headroom_min` | log GPU − log minimum GPU | 0 |
| 13 | `vram_headroom_gb` | VRAM − recommended VRAM | 0 |
| 14 | `spec_demand_index` | Weighted mix of recommended GPU, CPU, RAM, release year and genre | 0 |
| 15 | `headroom_x_resolution` | `gpu_headroom_rec` × (1 + 0.25 × max(0, `log_pixel_ratio`)) | 0 |
| 16 | `cpu_bottleneck_1080p` | log CPU / (1 + 0.35 × max(0, `log_pixel_ratio`)) | +1 |
| 17 | `vram_cliff_penalty` | VRAM shortfall against the recommendation, 35% stricter at 4K | 0 |
| 18 | `is_capped_60` | 1 when the title or genre suggests a 60 FPS engine lock | −1 |
| 19 | `genre_demand` | Genre load offset, −0.35 (esports) to +0.25 (AAA, sim) | −1 |
| 20 | `cpu_headroom_rec` | log CPU − log recommended CPU | +1 |

"""


def build_model_card(weights: dict[str, Any], eval_report: dict[str, Any]) -> str:
    """Render the Hugging Face model card for CanRun FPS Model v1.0."""
    summary = weights["training_summary"]
    cv_games = eval_report["leave_games_out_cv"]
    cv_gpus = eval_report["leave_gpus_out_cv"]
    m4_games = cv_games["M4_New_v9_TwoStage_LogRidge_Monotonic_GBDT"]
    m4_gpus = cv_gpus["M4_New_v9_TwoStage_LogRidge_Monotonic_GBDT"]
    n_trees = len(weights.get("gbdt_trees", []))
    n_features = len(weights.get("feature_names", []))

    front_matter = f"""---
license: apache-2.0
library_name: generic
pipeline_tag: tabular-regression
language:
  - en
tags:
  - fps-prediction
  - gaming
  - hardware-benchmarking
  - monotonic-gbdt
  - tabular-regression
  - nvidia-g-assist
  - pure-python
{_model_index(summary, m4_games, m4_gpus)}
---
"""

    header = f"""
# CanRun FPS Model v1.0

A small tabular model that estimates the native frame rate of a PC game at 1080p, 1440p or 4K from two things you can know before buying or installing: the PC's hardware scores and the game's Steam store requirements. It is the FPS engine of the [CanRun G-Assist plugin](https://mod.io/g/g-assist/m/canrun2#description) and runs with the Python standard library alone.

**Real-world accuracy:** on independent benchmarks the model alone is a median 18.9% off on recent games and NVIDIA cards, and 37% to 39% off on other test sets. The full CanRun app, which reads measured review curves first, is 9.3% off end to end, and about 10% off against a second reviewer on NVIDIA cards for games it has measurements for. Games without measurements still rely on this model, at 20% to 30%. No Steam player FPS data is used anywhere. Treat every prediction as a rough estimate, good for "will this run at 60 FPS?" and not for exact numbers.

**Try it:** the [CanRun Space](https://huggingface.co/spaces/grasant/canrun) runs this model alone. Pick a GPU, CPU, RAM and resolution, type a game, and it reads the live Steam store specs.

**Or just ask G-Assist.** In the [CanRun plugin](https://mod.io/g/g-assist/m/canrun2), press Alt+G and type "canrun cyberpunk 2077?". This is the full app's answer (review curves first, then this model), shortened, from a real run on an RTX 4090 with a 144 Hz screen:

```text
🎮 Game: Cyberpunk 2077
⚡ FPS: 113-144 (similar PC, native, RT off)
- Faster screen: about 113-170 FPS
- DLSS 4.5: DLAA, Quality to Performance
- Frame gen 2x: +115 FPS (usually 1.8x)
- Path tracing: -70 to -95 FPS (2-3x heavier)
🕹️ Steam Deck: Verified (30 FPS 800p) | Steam Frame: Playable
🎯 VERDICT: ✅🏃 CAN RUN
```

## Model details

| | |
| :--- | :--- |
| Developer | Bartosz Lenart ([CanRun](https://mod.io/g/g-assist/m/canrun2#description)) |
| Version | 1.0, released 2026 |
| Type | Ridge regression prior plus {n_trees} monotonic gradient-boosted trees, in log-FPS space |
| Inputs | {n_features} numeric features from GPU and CPU PassMark scores, VRAM, resolution and Steam minimum and recommended specs |
| Output | Native raster FPS (no upscaling, frame generation or ray tracing), clipped to 5 to 500 |
| Files | `model.json` (weights, about 920 KB) and `inference.py` (features and tree traversal, standard library only) |
| Speed | 0.06 ms median per prediction (0.12 ms p99), 6.5 ms to load `model.json`, measured on a Ryzen 7 7800X3D with Python 3.13. A full CanRun check takes 1 to 4 seconds, almost all of it Steam network requests |
| License | Apache-2.0 |

"""

    external = """## Real-world accuracy

In each test CanRun predicted the FPS for a game, a graphics card and a resolution, and the prediction was compared with the FPS a reviewer or player actually measured. "Model alone" is this model, called through `inference.py` with the game's store specs. "CanRun app" is the full plugin described in the next section.

What the columns mean:

- **Typical error** is the median miss: half the predictions were closer than this, half were further off. At 10%, a game predicted at 60 FPS typically ran within about 6 FPS of that.
- **Within 10% of real FPS** is the share of predictions that landed that close. For a prediction of 60 FPS, that means the real result was between about 55 and 67 FPS.
- **Real FPS inside the shown range.** The app shows a range, about 20% either side of its estimate. This column is how often the real result fell inside it.

TechSpot and OpenML were never used to build or tune CanRun. The RTX 5070 test is TechPowerUp's 2025 review, and the app's measured curves come from TechPowerUp's 2026 results, so that row compares the same reviewer one year apart.

| Test set | Points | Model alone: typical error | CanRun app: typical error | CanRun app: within 10% of real FPS | CanRun app: real FPS inside the shown range |
| :--- | :-: | :-: | :-: | :-: | :-: |
| TechPowerUp RTX 5070 review (March 2025), full app end to end: 24 games typed by name, 11 NVIDIA cards, 1080p and 1440p, live Steam requirements | 308 | n/a | **9.3%** | 53% | 71% |
| TechPowerUp, the 7 of those games without a CanRun benchmark curve, 19 GPUs | 266 | 18.9% | **18.7%** | 31% | n/a |
| TechSpot reviews (2025 to 2026): Forza Horizon 6, Crimson Desert, Battlefield 6, DOOM: The Dark Ages, on NVIDIA, AMD and Intel cards | 98 | 36.7% | **13.3%** (NVIDIA 10.0%) | 37% (NVIDIA 51%) | 69% (NVIDIA 82%) |
| [OpenML 42737 "fps-in-video-games"](https://www.openml.org/d/42737): UserBenchmark and FPS Benchmark user reports (2019 to 2020), 14 games, median per hardware group | 11,842 | 38.8% | **30.0%** | 17% | 35% |

How to read these numbers:

- **Thresholds.** In the end-to-end test the app put the game on the right side of 30 FPS 99.4% of the time and of 60 FPS 93.8% of the time.
- **Bias.** The model underestimates: by 14% on TechPowerUp, 35% on TechSpot and 8% on OpenML. The newest games are hit hardest.
- **Noise floor.** On OpenML a single user report differs from the median of identical hardware by 12.5%, so no predictor can get far below about 12% there.
- **Version mismatches.** OpenML results leave out CS:GO and the original Overwatch, because Steam now lists CS2 and Overwatch 2 requirements under those app IDs. With them, the app scores 33.8% and the model 45.9% (15,987 points).
- **Not scored.** 39 end-to-end cases (God of War Ragnarök, Ratchet & Clank: Rift Apart) found no Steam requirements from the typed title. 56 AMD cases were skipped because the G-Assist plugin accepts only NVIDIA GPUs.

All four tests can be rerun with `evaluate_demo_end_to_end.py`, `evaluate_external_benchmarks.py`, `evaluate_techspot_benchmarks.py` and `evaluate_openml_benchmarks.py` in `scripts/ds_pipeline/` of the CanRun repository.

### How the CanRun app improves on the model

This model is unchanged since v1.0. CanRun 12.2 wraps it with measured review curves, laptop GPU scores and an engine correction. On the same tests (median error, lower is better):

| Test | This model alone | Full CanRun 12.2 app |
| :--- | :---: | :---: |
| TechSpot desktops, NVIDIA cards (55 points) | 33.6% | **10.0%** |
| TechSpot desktops, all cards (98) | 36.7% | **13.3%** |
| TechSpot desktops, AMD and Intel cards (43) | 40.7% | **16.2%** |
| OpenML user reports, older games (11,842) | 38.8% | **30.0%** |
| Games without review data, each measured game held out (30 games, NVIDIA) | 25.0% | **23.1%** |
| TechPowerUp 2025 games without a curve (266) | 18.9% | **18.7%** |

Where the app has review data for a game, it is about three times closer than the model alone, and on TechSpot's NVIDIA results the real FPS fell inside the app's shown range (about ±20%) 82% of the time. On games without review data the app is close to the model, because the model makes that estimate; the engine correction adds about 2 points. The app was also scored on cases the model cannot take on its own: Notebookcheck laptops at 14.9% (12.1, which read laptop GPUs as desktop cards, was 23.0% and 39% too high), and TechPowerUp's RTX 5070 review end to end, games typed by name, at 9.3%.

1. **Live Steam requirements.** The app looks up the typed title in the Steam store when the check runs and maps the minimum and recommended GPU, CPU and RAM to PassMark scores, so a game can be checked on release day.
2. **Measured curves first.** For 33 games the app has review measurements: 32 from TechPowerUp (the June 2026 suite with 21 cards, and per-game reviews with 30 to 43 cards down to the RTX 3050) at three resolutions, and ARC Raiders from ComputerBase's 2026 suite at 1440p and 4K. They ship inside the CanRun plugin and are not published in this repository. The app reads that game's curve at the user's GPU score, fitted on the user's GPU vendor. Scored against TechSpot's separate measurements of four of those games, this cut the app's error from 32.9% to 13.3%, and to 10.0% on NVIDIA cards. A card left out of its own curve is predicted within 6.0% (NVIDIA), 8.3% for NVIDIA cards below an RTX 3070, and 9.4% (AMD). On Notebookcheck's laptop results the app is 14.9% off, where 12.1 read laptop GPUs as desktop cards and was 39% too high. Other games fall back to the 79 curves in CanRun's older benchmark table that carry outside information; on OpenML that lowers the error from 38.8% to 30.2%. The table's 1,618 store-spec curves are skipped, so those games use this model directly. In the end-to-end test the curve path scored 6.9% on 176 points.
3. **The model for everything else.** Games without a curve use this model: 18.5% on 132 end-to-end points. CanRun 12.2 multiplies its output by an engine factor for Unreal Engine 5 (×0.77) and RE Engine (×1.11), with engines taken from Wikidata; holding out each of 30 measured games, that lowers the error from 25.0% to 23.1% on NVIDIA cards.
4. **Matching the right game.** Whole-word title matching cut the end-to-end error from 35.1% to 15.2% with no model change, before the measured curves were added. The old matcher sent "Assassin's Creed Mirage" to "RAGE".

The app's store-spec path passes the model both GPU scores, the user's VRAM, the recommended CPU and RAM from the store page, and Steam's release year and genres. When it passed only the two GPU scores, it scored 20.8% on those 7 games and was 54% off on F1 24. With the full specs it scores 18.7%, and F1 24 is 29% off.

**How the store specs are used.** 8 of the 21 features come from the Minimum and Recommended specs: both GPU scores and the user's headroom over each, VRAM headroom and a VRAM cliff penalty, a demand index from the recommended GPU, CPU and RAM, and CPU headroom over the recommended CPU. A page with only one GPU gets the other at a 1.65× ratio. The same specs also generated most training targets (see Training data), so the model mostly learned how far a PC sits above the recommended GPU.

### Compared with other FPS estimators

| Tool | Published accuracy | Test data |
| :--- | :--- | :--- |
| **CanRun 12.2** | 10.0% median error (NVIDIA desktops), 13.3% (all desktop cards), 14.9% (laptops), 9.3% (end to end), 30.0% (older games, user reports) | TechSpot, Notebookcheck and OpenML were never used to build it. TechPowerUp's 2026 results are its measured curves; the 9.3% test is TechPowerUp's 2025 RTX 5070 review |
| **CanRun 12.2, game nobody has measured** | 18.7% to 23.1% median error | TechPowerUp games without a curve, and each measured game held out; store-spec model only |
| [MyRigCheck](https://myrigcheck.games/accuracy), TechSavant | about 15% (claimed) | No public test set |
| [PCBottleneck](https://www.pcbottleneck.com/methodology) | 21.6% mean error | Its own estimate path |
| [FPSGauge](https://fpsgauge.com/methodology) | 26% median error | 1,990 independent results |
| [FPSHQ](https://fpshq.com/en/methodology/) | 41.6% median error | 1,341 independent review results |
| [SteamFPS](https://steamfps.com/methodology) | None published | Measured FPS for 239 Steam Deck games; other figures are estimated or derived |
| Ramkumar et al. 2025 | R² 0.99 | OpenML 42737, scored on its own training data |

Each tool was tested on different data, so the table compares the size of each error. On the one shared test, 16 new 2025 and 2026 games at 1080p on an RTX 4060 (the class SteamFPS uses for the Steam Machine), CanRun was 8.0% off with measured curves and 10.7% from store specs alone; SteamFPS gives only a color rating for those games.

"""

    uses = """## Intended use

- A quick, local "can my PC run this game, and roughly how fast?" check before buying or installing, on desktop PCs with NVIDIA GeForce GTX or RTX cards.
- An open baseline for FPS-estimation research: small, inspectable and dependency-free.

## Out of scope

- **Exact FPS or purchase guarantees.** Expect errors of 15% to 40%. Use it to compare options or check a threshold, not as a promise.
- **Laptops.** Laptop GPUs run at lower power than the desktop cards they share names with, and no laptop data was tested.
- **Upscaling, frame generation and ray tracing.** The output is native raster FPS. The CanRun plugin reports DLSS and path tracing separately as rough multipliers.
- **Games with misleading store specs.** Predictions are only as good as the publisher's minimum and recommended listings. Store specs that are missing, outdated or copied between games produce wrong estimates.
- **Stutter, shader compilation, driver bugs and 1% lows.** None of these can be predicted from store specs.

## Bias and known failure modes

- **Newer games are underestimated.** On 2025 and 2026 TechSpot reviews the app is about 30% low, and Forza Horizon 6 at High is 56% low.
- **Old, CPU-limited PCs are overestimated.** On OpenML the app averages +22%, with Rainbow Six Siege at +139% and ARK at +116%.
- **AMD and Intel GPUs are less accurate.** On TechSpot the model is 40.7% off on AMD and Intel cards against 33.6% on NVIDIA, because the training curves come mostly from NVIDIA cards.
- **60 FPS locks are guessed from the title and genre.** A locked game the rule misses can be predicted far above 60.
- **CPU scores are coarse.** CPUs map to PassMark through family patterns, so two CPUs in one family can get the same score.

"""

    quickstart = f"""## Quickstart

```python
import importlib.util
from huggingface_hub import hf_hub_download

inference_path = hf_hub_download(repo_id="{DEFAULT_REPO_ID}", filename="inference.py")
model_path = hf_hub_download(repo_id="{DEFAULT_REPO_ID}", filename="model.json")

spec = importlib.util.spec_from_file_location("canrun_inference", inference_path)
canrun_inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canrun_inference)

model = canrun_inference.CanRunFPSModel.from_pretrained(model_path)

# Cyberpunk 2077 at 1440p on an RTX 4070 with a Ryzen 7 7800X3D
result = model.predict_with_interval(
    gpu_name="RTX 4070",
    gpu_passmark=26950,
    gpu_vram_gb=12,
    cpu_passmark=34300,
    resolution="1440p",
    rec_gpu_passmark=19500,   # recommended: RTX 2060 SUPER class
    min_gpu_passmark=10100,   # minimum: GTX 1060 6GB
    rec_cpu_passmark=17500,
    rec_vram_gb=8,
    rec_ram_gb=16,
    release_year=2020,
    genre="Action RPG Open World Cyberpunk",
    game_name="Cyberpunk 2077",
)
print(result)
# {{'expected_fps': 60.93, 'conformal_80_low': 57.8, 'conformal_80_high': 64.3, 'resolution': '1440p', 'model_version': '1.0'}}
# The interval reflects the training suite only; real error is often ±30%.
```

`inference.py` also has `fetch_steam_store_metadata(app_id)`, which reads the release year, genre and name from Steam's public store API, and `predict_fps(...)`, which returns the number alone. A title the 60 FPS rule recognizes is capped:

```python
model.predict_fps(gpu_name="RTX 4090", gpu_passmark=38900, gpu_vram_gb=24, cpu_passmark=34300,
                  resolution="4K", rec_gpu_passmark=14500, min_gpu_passmark=8500,
                  game_name="Elden Ring", genre="Action RPG")
# 60.0
```

"""

    training = f"""## Training data

**The training targets are mostly generated, not measured.** Each of the {summary['unique_games']:,} games gets an FPS curve over {summary['unique_gpus']} GPUs and 3 resolutions from a power law anchored at the RTX 4090:

$$\\text{{FPS}}(\\text{{GPU}}) = \\text{{anchor}}_{{4090}} \\cdot \\left(\\frac{{\\text{{PassMark}}_{{\\text{{GPU}}}}}}{{38{{,}}192}}\\right)^{{\\beta}}$$

The anchor and the exponent β, set per resolution, come from one of three sources:

| Source | Games | Where the anchor and exponent come from |
| :--- | :-: | :--- |
| Measured benchmarks | 21 | Published benchmark results, with missing GPUs and resolutions filled in by the formula |
| Hand-calibrated catalog | 58 | Set by hand per game (RTX 4090 FPS and scaling exponent) |
| Steam store specs | 628 | A fixed rule on the recommended GPU's PassMark score, targeting roughly 68 FPS at 1080p on the recommended GPU |
| Placeholder specs by release year | 990 | The same rule, on minimum and recommended specs picked from the release year (for example GTX 1060 6GB recommended for 2018 to 2021), because no store specs were parsed |

Special-case rules cover 60 FPS locked games and light indie games capped at 240 FPS. The {summary['primary_benchmark_curves']:,} curves are then expanded over {summary['unique_cpus']} CPU tiers (Ryzen 5 3600 to Ryzen 7 7800X3D) with a CPU-bottleneck factor, giving {summary['total_records']:,} rows. Store specs come from Steam's public store API, and game discovery uses Steam Web API chart endpoints with the developer's own key.

**No Steam player FPS.** The pipeline also asks Steam's `IGamePerformanceStatsService/GetGameFrameRateStats` for player frame rates per hardware cluster. That endpoint is not in Valve's [Web API reference](https://partner.steamgames.com/doc/webapi) and returns HTTP 401 to outside keys, so the dataset contains 0 Steam FPS reports. The only measured frame rates behind training are the published benchmarks for 21 games.

**What this means for the scores.** For about 95% of games the target is a fixed function of the same store-spec features the model receives, and for 990 of them those specs are placeholders too. The cross-validation below therefore mostly measures how well the model reproduces that rule. That is why it reports about 1% error while independent benchmarks show 15% to 40%. Use the real-world numbers above.

Four automated checks run before training: removal of curves copied from a generic template (143 repaired), FPS that falls from 1080p to 1440p to 4K for every game and GPU, robust repair of single-card outliers per game and resolution (6 repaired), and full coverage of resolutions and CPU tiers.

**Not published.** The benchmark table, the {summary['primary_benchmark_curves']:,}-entry lookup table and the harvested Steam store data stay private, to respect the [Steam Web API Terms of Use](https://steamcommunity.com/dev/apiterms). This repository holds only learned weights and code. To rebuild a dataset, use `scripts/ds_pipeline/collect_and_audit.py` in the CanRun repository with your own [Steam Web API key](https://steamcommunity.com/dev/apikey).

## Internal cross-validation

5-fold `GroupKFold` on the synthetic suite above. This checks that the model learned the generating rule without memorizing games or GPUs. It is not a measure of real-world accuracy.

**Unseen games** (grouped by game):

| Model | MdAPE | MAPE | R² | MAE (FPS) | Monotonicity violations |
| :--- | :-: | :-: | :-: | :-: | :-: |
{_cv_rows(cv_games)}

**Unseen GPUs** (grouped by GPU):

| Model | MdAPE | MAPE | R² | MAE (FPS) | Monotonicity violations |
| :--- | :-: | :-: | :-: | :-: | :-: |
{_cv_rows(cv_gpus)}

The legacy v8.0 model was a seven-model stacking ensemble trained on 179 rows from 15 games, with one column per game and no store requirements. It scored R² 0.984 on a random split, and its shipped lookup table scored R² 0.0932 on unseen games. This model gets no game or GPU identifiers, which is why it holds up when grouped.

Running the v8.0 ensemble itself on the same independent benchmarks (`scripts/ds_pipeline/evaluate_legacy_v8.py`), the median error was 79.8% on TechPowerUp (this model: 18.9%), 75.1% on TechSpot (36.7%) and 39.8% on OpenML (38.8%). v8.0 overestimated new games by about +90% to +110%. On older popular games the two are roughly tied. The simple spec rule scored 19.8%, 37.5% and 36.7% on the same three tests, so on OpenML it beat both models on the median. This model still makes fewer large misses than the rule: on TechPowerUp its mean error is 18.8% against 23.1%, its average miss 26.5 FPS against 34.4 FPS, and R² 0.503 against 0.064. On OpenML its mean error is 43.1% against 47.7%. It does better because it also reads the minimum GPU, CPU, VRAM, RAM, release year and genre, which it learned to weigh from the 79 measured and hand-calibrated games (F1 24: 29.2% off against the rule's 54.4%). It also gives a range and stays monotonic.

"""

    closing = f"""## Why plain JSON

The weights are plain JSON, and `inference.py` (about 420 lines of standard-library Python) computes the features and walks the trees. Nothing is unpickled, so loading the model cannot run code. It also installs into the G-Assist plugin without `scikit-learn`, `numpy` or `onnxruntime`. The staging script checks that `inference.py` gives the same features and predictions as the CanRun runtime on a set of probe cases.

## Terms and trademarks

CanRun and this model are independent and not affiliated with or endorsed by Valve or NVIDIA. No Steam user data or API keys are collected or stored. Steam store data fetched at run time, and predictions derived from it, are provided as is, without warranty. Valve and Steam are trademarks of Valve Corporation. NVIDIA, GeForce, RTX, DLSS and G-Assist are trademarks of NVIDIA Corporation.

## Citation

```bibtex
@misc{{canrun_fps_model_v1_2026,
  author       = {{Lenart, Bartosz}},
  title        = {{CanRun FPS Model v1.0: A Two-Stage Ridge and Monotonic GBDT Model for PC Game Frame-Rate Estimation}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{DEFAULT_REPO_ID}}}}}
}}
```

## Contact

Questions, wrong predictions and benchmark data are welcome in the [Community tab](https://huggingface.co/{DEFAULT_REPO_ID}/discussions).
"""

    return front_matter + header + external + uses + quickstart + training + _ARCHITECTURE_MD + _FEATURES_MD + closing
