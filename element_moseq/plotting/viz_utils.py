# ---- Modified version of the viz functions from the main branch of keypoint_moseq  ----

import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Optional, Tuple

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np

logger = dj.logger

_DLC_SUFFIX_RE = re.compile(
    r"(?:DLC_[A-Za-z0-9]+[A-Za-z]+(?:\d+)?(?:[A-Za-z]+)?"  # scorer-ish token
    r"(?:\w+)*)"  # optional extra blobs
    r"(?:shuffle\d+)?"  # shuffleN
    r"(?:_\d+)?$"  # _iter
)


def _normalize_name(name: str) -> str:
    """
    Normalize a recording/video string for matching:
    - lowercase, strip whitespace
    - drop extension if present
    - remove common DLC suffix blob (e.g., '...DLC_resnet50_...shuffle1_500000')
    - collapse separators to single spaces
    """
    s = name.lower().strip()
    s = Path(s).stem
    s = _DLC_SUFFIX_RE.sub("", s)  # strip DLC tail if present
    s = re.sub(r"[\s._-]+", " ", s).strip()
    return s


def build_recording_to_video_id(
    recording_names: List[str],
    video_paths: List[str],
    video_ids: List[int],
    fuzzy_threshold: float = 0.80,
) -> Dict[str, Optional[int]]:
    """
    Returns: {recording_name -> video_id or None if no good match}
    Strategy: exact normalized stem match; if none, substring; then fuzzy.
    """
    # candidate stems from videos
    stems: List[Tuple[str, int]] = [
        (_normalize_name(Path(p).name), vid) for p, vid in zip(video_paths, video_ids)
    ]

    mapping: Dict[str, Optional[int]] = {}

    for rec in recording_names:
        nrec = _normalize_name(rec)

        # 1) exact normalized match
        exact = [vid for stem, vid in stems if stem == nrec]
        if exact:
            mapping[rec] = exact[0]
            continue

        # 2) substring either way (choose longest stem to disambiguate)
        subs = [(stem, vid) for stem, vid in stems if nrec in stem or stem in nrec]
        if subs:
            subs.sort(key=lambda x: len(x[0]), reverse=True)
            mapping[rec] = subs[0][1]
            continue

        # 3) fuzzy best match
        best_vid, best_ratio = None, 0.0
        for stem, vid in stems:
            r = SequenceMatcher(None, nrec, stem).ratio()
            if r > best_ratio:
                best_ratio, best_vid = r, vid
        mapping[rec] = best_vid if best_ratio >= fuzzy_threshold else None

    return mapping


def plot_medoid_distance_outliers(
    project_dir: str,
    recording_name: str,
    original_coordinates: np.ndarray,
    interpolated_coordinates: np.ndarray,
    outlier_mask,
    outlier_thresholds,
    bodyparts: list[str],
    **kwargs,
):
    """Create and save a plot comparing distance-to-medoid for original vs. interpolated keypoints.

    Generates a multi-panel plot showing the distance from each keypoint to the medoid
    position for both original and interpolated coordinates. The plot includes threshold
    lines and shaded regions for outlier frames. Saves the figure to the QA plots
    directory.

    Parameters
    -------
    project_dir: str
        Path to the project directory where the plot will be saved.

    recording_name: str
        Name of the recording, used for the plot title and filename.

    original_coordinates: ndarray of shape (n_frames, n_keypoints, keypoint_dim)
        Original keypoint coordinates before interpolation.

    interpolated_coordinates: ndarray of shape (n_frames, n_keypoints, keypoint_dim)
        Keypoint coordinates after interpolation.

    outlier_mask: ndarray of shape (n_frames, n_keypoints)
        Boolean mask indicating outlier keypoints (True = outlier).

    outlier_thresholds: ndarray of shape (n_keypoints,)
        Distance thresholds for each keypoint above which points are considered outliers.

    bodyparts: list of str
        Names of bodyparts corresponding to each keypoint. Must have length equal to
        n_keypoints.

    **kwargs
        Additional keyword arguments (ignored), usually overflow from **config().

    Returns
    -------
    None
        The plot is saved to 'QA/plots/keypoint_distance_outliers/{recording_name}.png'.
    """
    from keypoint_moseq.util import get_distance_to_medoid, plot_keypoint_traces

    plot_path = os.path.join(
        project_dir,
        "quality_assurance",
        "plots",
        "keypoint_distance_outliers",
        f"{recording_name}.png",
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    original_distances = get_distance_to_medoid(
        original_coordinates
    )  # (n_frames, n_keypoints)
    interpolated_distances = get_distance_to_medoid(
        interpolated_coordinates
    )  # (n_frames, n_keypoints)

    fig = plot_keypoint_traces(
        traces=[original_distances, interpolated_distances],
        plot_title=recording_name,
        bodyparts=bodyparts,
        line_labels=["Original", "Interpolated"],
        thresholds=outlier_thresholds,
        shading_mask=outlier_mask,
    )

    fig.savefig(plot_path, dpi=300)

    plt.close()
    logger.info(
        f"Saved keypoint distance outlier plot for {recording_name} to {plot_path}."
    )
    return fig


def plot_pcs(
    pca,
    *,
    use_bodyparts,
    skeleton,
    keypoint_colormap="autumn",
    keypoint_colors=None,
    savefig=True,
    project_dir=None,
    scale=1,
    plot_n_pcs=10,
    axis_size=(2, 1.5),
    ncols=5,
    node_size=30.0,
    line_width=2.0,
    interactive=True,
    **kwargs,
):
    """
    Visualize the components of a fitted PCA model.

    For each PC, a subplot shows the mean pose (semi-transparent) along with a
    perturbation of the mean pose in the direction of the PC.

    Parameters
    ----------
    pca : :py:func:`sklearn.decomposition.PCA`
        Fitted PCA model

    use_bodyparts : list of str
        List of bodyparts to that are used in the model; used to index bodypart
        names in the skeleton.

    skeleton : list
        List of edges that define the skeleton, where each edge is a pair of
        bodypart names.

    keypoint_colormap : str
        Name of a matplotlib colormap to use for coloring the keypoints.

    keypoint_colors : array-like, shape=(num_keypoints,3), default=None
        Color for each keypoint. If None, `keypoint_colormap` is used. If the
        dtype is int, the values are assumed to be in the range 0-255,
        otherwise they are assumed to be in the range 0-1.

    savefig : bool, True
        Whether to save the figure to a file. If true, the figure is saved to
        `{project_dir}/pcs-{xy/xz/yz}.pdf` (`xz` and `yz` are only included
        for 3D data).

    project_dir : str, default=None
        Path to the project directory. Required if `savefig` is True.

    scale : float, default=0.5
        Scale factor for the perturbation of the mean pose.

    plot_n_pcs : int, default=10
        Number of PCs to plot.

    axis_size : tuple of float, default=(2,1.5)
        Size of each subplot in inches.

    ncols : int, default=5
        Number of columns in the figure.

    node_size : float, default=30.0
        Size of the keypoints in the figure.

    line_width: float, default=2.0
        Width of edges in skeleton

    interactive : bool, default=True
        For 3D data, whether to generate an interactive 3D plot.
    """
    from jax_moseq.models.keypoint_slds import center_embedding
    from keypoint_moseq.util import get_edges
    from keypoint_moseq.viz import plot_pcs_3D

    k = len(use_bodyparts)
    d = len(pca.mean_) // (k - 1)

    if keypoint_colors is None:
        cmap = plt.cm.get_cmap(keypoint_colormap)
        keypoint_colors = cmap(np.linspace(0, 1, k))

    Gamma = np.array(center_embedding(k))
    edges = get_edges(use_bodyparts, skeleton)
    plot_n_pcs = min(plot_n_pcs, pca.components_.shape[0])

    magnitude = np.sqrt((pca.mean_**2).mean()) * scale
    ymean = Gamma @ pca.mean_.reshape(k - 1, d)
    ypcs = (pca.mean_ + magnitude * pca.components_).reshape(-1, k - 1, d)
    ypcs = Gamma[np.newaxis] @ ypcs[:plot_n_pcs]

    if d == 2:
        dims_list, names = [[0, 1]], ["xy"]
    if d == 3:
        dims_list, names = [[0, 1], [0, 2]], ["xy", "xz"]

    for dims, name in zip(dims_list, names):
        nrows = int(np.ceil(plot_n_pcs / ncols))
        fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        for i, ax in enumerate(axs.flat):
            if i >= plot_n_pcs:
                ax.axis("off")
                continue

            for e in edges:
                ax.plot(
                    *ymean[:, dims][e].T,
                    color=keypoint_colors[e[0]],
                    zorder=0,
                    alpha=0.25,
                    linewidth=line_width,
                )
                ax.plot(
                    *ypcs[i][:, dims][e].T,
                    color="k",
                    zorder=2,
                    linewidth=line_width + 0.2,
                )
                ax.plot(
                    *ypcs[i][:, dims][e].T,
                    color=keypoint_colors[e[0]],
                    zorder=3,
                    linewidth=line_width,
                )

            ax.scatter(
                *ymean[:, dims].T,
                c=keypoint_colors,
                s=node_size,
                zorder=1,
                alpha=0.25,
                linewidth=0,
            )
            ax.scatter(
                *ypcs[i][:, dims].T,
                c=keypoint_colors,
                s=node_size,
                zorder=4,
                edgecolor="k",
                linewidth=0.2,
            )

            ax.set_title(f"PC {i+1}", fontsize=10)
            ax.set_aspect("equal")
            ax.axis("off")

        fig.set_size_inches((axis_size[0] * ncols, axis_size[1] * nrows))
        plt.tight_layout()

        if savefig:
            assert project_dir is not None, fill(
                "The `savefig` option requires a `project_dir`"
            )
            plt.savefig(os.path.join(project_dir, f"pcs-{name}.pdf"))
        plt.show()

    if interactive and d == 3:
        plot_pcs_3D(
            ymean,
            ypcs,
            edges,
            keypoint_colormap,
            project_dir if savefig else None,
            node_size / 3,
            line_width * 2,
        )
    return fig


def copy_pdf_to_png(project_dir, model_name):
    """
    Convert PDF progress plot to PNG format using pdf2image.
    The fit_model function generates a single fitting_progress.pdf file.
    This function should always succeed if the PDF exists.

    Args:
        project_dir (Path): Project directory path
        model_name (str): Model name directory

    Returns:
        bool: True if conversion was successful, False otherwise

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        RuntimeError: If conversion fails
    """
    from pdf2image import convert_from_path

    # Construct paths for PDF and PNG files
    model_dir = Path(project_dir) / model_name
    pdf_path = model_dir / "fitting_progress.pdf"
    png_path = model_dir / "fitting_progress.png"

    # Check if PDF exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF progress plot not found at {pdf_path}")

    # Convert PDF to PNG
    images = convert_from_path(pdf_path, dpi=300)
    if not images:
        raise ValueError(f"No PDF file found at {pdf_path}")

    images[0].save(png_path, "PNG")
    logger.info(f"Generated PNG progress plot at {png_path}")
    return True
