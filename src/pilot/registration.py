"""Warp a photograph of a printed film back onto the digital image it came from.

Everything downstream of the pilot needs this. Comparing a photograph to a
synthetic degradation of its source is only meaningful if the two are in the
same frame -- otherwise every statistic is contaminated by the desk, the
lightbox bezel, and however far away the photographer stood. A paired GAN arm
(`docs/phase2_data_notes.md`) needs it even more strictly: unpaired training
is the fallback you take when registration fails, not the plan.

The approach is deliberately boring: ORB features, ratio-tested matches,
RANSAC homography. A printed radiograph photographed roughly flat is close to
a planar scene under projective transform, which is exactly the case a
homography is exact for. Lens distortion is the residual, and it shows up as a
reprojection error that grows toward the frame edges -- `RegistrationResult`
reports the error rather than hiding it, because a pilot with 8-pixel
registration error cannot support a claim about high-frequency detail.

Contrast handling: a photographed film is usually low-contrast and unevenly
lit compared to its source, and ORB on raw pixels finds few durable
correspondences. CLAHE on both sides before detection is what makes this work
in practice; it is a matching aid only and never touches the returned pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Working resolution for feature detection. Detecting at full phone resolution
# is slower and finds features in print grain rather than anatomy.
_WORK_LONG_EDGE = 1024


@dataclass
class RegistrationResult:
    """The homography, the warped photograph, and how much to trust them."""

    ok: bool
    #: Photo warped into the reference's frame, same shape as the reference.
    #: None when registration failed -- callers must check `ok`, not truthiness
    #: of the array.
    warped: np.ndarray | None
    #: 3x3 homography mapping photo pixels -> reference pixels.
    homography: np.ndarray | None
    n_matches: int
    n_inliers: int
    #: Mean RANSAC-inlier reprojection error, in reference pixels.
    reprojection_error: float
    reason: str = ""

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_matches if self.n_matches else 0.0


def _prep(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Grayscale + CLAHE + downscale for detection. Returns (image, scale)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    scale = min(1.0, _WORK_LONG_EDGE / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray), scale


def _quadrilateral_is_sane(h_matrix: np.ndarray, shape: tuple[int, int]) -> bool:
    """Reject homographies that fold, mirror, or collapse the photograph.

    RANSAC will happily return a degenerate matrix when the matches are noise,
    and the warp it produces looks like plausible garbage. Checking that the
    photo's corners land as a convex, positively-oriented, non-tiny quad is
    cheap and catches that case before it becomes a statistic.
    """
    h, w = shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, h_matrix).reshape(4, 2)
    if not np.isfinite(mapped).all():
        return False
    if cv2.contourArea(mapped.astype(np.float32)) < 0.01 * w * h:
        return False
    if not cv2.isContourConvex(mapped.astype(np.float32)):
        return False
    # Positive area under OpenCV's ordering means orientation was preserved;
    # a mirrored film is a bug in the pipeline, not a capture condition.
    signed = cv2.contourArea(mapped.astype(np.float32), oriented=True)
    return signed > 0


def register_photo(
    photo: np.ndarray,
    reference: np.ndarray,
    min_inliers: int = 25,
    max_reprojection_error: float = 6.0,
    n_features: int = 4000,
) -> RegistrationResult:
    """Estimate the homography taking `photo` onto `reference` and apply it.

    Args:
        photo: BGR uint8 photograph of the printed film, background and all.
        reference: BGR uint8 source radiograph the film was printed from.
        min_inliers: below this, the match is called a failure rather than
            returned with a caveat. 25 is loose enough for a badly glared or
            heavily blurred capture and tight enough to reject noise.
        max_reprojection_error: mean inlier error, in reference pixels, above
            which the result is marked not ok. A pilot statistic about
            high-frequency detail is meaningless at several pixels of
            misalignment.
        n_features: ORB budget per image.

    Returns:
        A `RegistrationResult`. Failure is reported, never raised: a pilot set
        will contain a few unregisterable frames (severe glare erases the
        features it covers), and the report counts them instead of stopping.
    """
    if photo.ndim != 3 or reference.ndim != 3:
        raise ValueError("register_photo expects BGR images; got "
                         f"photo.ndim={photo.ndim}, reference.ndim={reference.ndim}")

    photo_gray, photo_scale = _prep(photo)
    ref_gray, ref_scale = _prep(reference)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp_photo, des_photo = orb.detectAndCompute(photo_gray, None)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    if des_photo is None or des_ref is None or len(kp_photo) < 4 or len(kp_ref) < 4:
        return RegistrationResult(False, None, None, 0, 0, float("nan"),
                                  "too few ORB features to match")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(des_photo, des_ref, k=2)
    # Lowe's ratio test. Panoramic radiographs are near-symmetric left/right,
    # which produces exactly the ambiguous second-best match this rejects.
    matches = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(matches) < 4:
        return RegistrationResult(False, None, None, len(matches), 0, float("nan"),
                                  "fewer than 4 ratio-test matches")

    src = np.float32([kp_photo[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    h_work, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=5000)
    if h_work is None:
        return RegistrationResult(False, None, None, len(matches), 0, float("nan"),
                                  "RANSAC found no homography")

    inlier_mask = mask.ravel().astype(bool)
    n_inliers = int(inlier_mask.sum())
    projected = cv2.perspectiveTransform(src[inlier_mask], h_work).reshape(-1, 2)
    err_work = float(np.mean(np.linalg.norm(projected - dst[inlier_mask].reshape(-1, 2), axis=1)))

    # Undo the detection-time downscaling on both sides so the homography maps
    # full-resolution photo pixels to full-resolution reference pixels.
    to_work = np.diag([photo_scale, photo_scale, 1.0]).astype(np.float64)
    from_work = np.diag([1.0 / ref_scale, 1.0 / ref_scale, 1.0]).astype(np.float64)
    homography = from_work @ h_work @ to_work
    err = err_work / ref_scale

    if not _quadrilateral_is_sane(homography, photo.shape[:2]):
        return RegistrationResult(False, None, homography, len(matches), n_inliers, err,
                                  "degenerate homography (folded or collapsed quad)")

    ref_h, ref_w = reference.shape[:2]
    warped = cv2.warpPerspective(photo, homography, (ref_w, ref_h))

    reason = ""
    ok = True
    if n_inliers < min_inliers:
        ok, reason = False, f"only {n_inliers} inliers (need {min_inliers})"
    elif err > max_reprojection_error:
        ok, reason = False, f"reprojection error {err:.2f}px > {max_reprojection_error}px"
    return RegistrationResult(ok, warped, homography, len(matches), n_inliers, err, reason)
