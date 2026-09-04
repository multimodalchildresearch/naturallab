#!/usr/bin/env python
"""Strict, transactional extraction of NaturalLab XDF recordings."""

import os
import sys
import argparse
import base64
import hashlib
import json
import re
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
import cv2

try:
    import pyxdf
except ImportError:
    pyxdf = None

try:
    from tqdm import tqdm
except ImportError as error:
    raise ImportError(
        "tqdm is required; install NaturalLab's core dependencies"
    ) from error


def _stream_name(stream):
    """Return an XDF stream name without exposing pyxdf's list wrapping."""
    try:
        return str(stream["info"]["name"][0])
    except (KeyError, IndexError, TypeError):
        return "<unknown>"


def _stream_type(stream):
    """Return an XDF stream type without exposing pyxdf's list wrapping."""
    try:
        return str(stream["info"]["type"][0])
    except (KeyError, IndexError, TypeError):
        return ""


def _is_declared_imu_stream(stream):
    """Return whether stream metadata explicitly identifies an IMU stream."""
    name = _stream_name(stream).lower()
    return _stream_type(stream).lower() == "imu" or name in {
        "neonimu",
        "neon_imu",
        "imu",
    }


def _safe_filename_component(value):
    """Make a deterministic, portable filename component."""
    original = str(value).strip()
    component = re.sub(r"[^a-z0-9]+", "_", original.lower()).strip("_")
    component = component or "stream"
    if len(component) > 80:
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
        component = f"{component[:67].rstrip('_')}_{digest}"
    if component in {"con", "prn", "aux", "nul"}:
        component = f"stream_{component}"
    return component


def _stream_output_paths(stream, stem):
    """Return every relative path reserved by one extracted stream."""

    stream_type = _stream_type(stream).strip().lower()
    if _is_declared_imu_stream(stream):
        return {f"{stem}.csv"}
    if stream_type == "videostream":
        return {f"{stem}.mp4", f"{stem}_timestamps.csv"}
    if stream_type == "audio":
        return {f"{stem}.wav", f"{stem}_timestamps.csv"}
    if stream_type in {"depth", "depthdata"}:
        return {
            f"{stem}_depth",
            f"{stem}_visualization.mp4",
            f"{stem}_timestamps.csv",
            f"{stem}_depth_metadata.json",
        }
    if stream_type == "deviceinfo":
        return {f"{stem}.json"}
    return {f"{stem}.csv"}


def _plan_stream_output_stems(streams):
    """Plan safe unique stems and reject every output-path ambiguity."""

    stem_owners = {}
    path_owners = {}
    planned = {}
    for index, stream in enumerate(streams):
        original_name = _stream_name(stream)
        if original_name == "<unknown>" or not original_name.strip():
            raise RuntimeError(f"XDF stream {index} has no usable name")
        stem = _safe_filename_component(original_name)
        existing_stem = stem_owners.get(stem.casefold())
        if existing_stem is not None:
            raise RuntimeError(
                "ambiguous XDF stream names map to the same output stem "
                f"{stem!r}: {existing_stem!r} and {original_name!r}"
            )
        stem_owners[stem.casefold()] = original_name
        for relative_path in _stream_output_paths(stream, stem):
            key = relative_path.casefold()
            existing_path = path_owners.get(key)
            if existing_path is not None:
                raise RuntimeError(
                    "ambiguous XDF streams would share output path "
                    f"{relative_path!r}: {existing_path!r} and {original_name!r}"
                )
            path_owners[key] = original_name
        planned[id(stream)] = stem
    return planned


def _resolved_output_stem(stream_name, output_stem=None):
    """Return a safe stem even for direct extractor calls."""

    return _safe_filename_component(
        stream_name if output_stem is None else output_stem
    )


def _prepare_extraction_staging_dir(output_dir):
    """Validate the final target and create a same-filesystem staging tree."""

    final_output_dir = os.path.abspath(os.fspath(output_dir))
    if os.path.lexists(final_output_dir):
        if os.path.islink(final_output_dir) or not os.path.isdir(final_output_dir):
            raise RuntimeError(
                "XDF output path is not a normal directory: "
                f"{final_output_dir}"
            )
        with os.scandir(final_output_dir) as entries:
            if next(entries, None) is not None:
                raise RuntimeError(
                    "XDF output directory is not empty: "
                    f"{final_output_dir}. "
                    "Choose a new empty directory so stale files cannot be "
                    "mistaken for this extraction."
                )
    parent_dir = os.path.dirname(final_output_dir)
    os.makedirs(parent_dir, exist_ok=True)
    staging_dir = tempfile.mkdtemp(
        prefix=f".{os.path.basename(final_output_dir)}.staging-",
        dir=parent_dir,
    )
    return final_output_dir, staging_dir


def _publish_extraction_staging_dir(staging_dir, final_output_dir):
    """Atomically publish a complete staging directory over an empty target."""

    os.replace(staging_dir, final_output_dir)


def _unwrap_singleton(value):
    """Unwrap list/array containers used by pyxdf for scalar metadata."""
    while True:
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0]
            continue
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
            continue
        return value


def _declared_channel_count(stream, stream_name):
    """Return one positive channel count from XDF metadata."""

    try:
        raw_count = _unwrap_singleton(stream["info"]["channel_count"])
        channel_count = int(raw_count)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"invalid declared channel count for {stream_name}"
        ) from error
    if channel_count <= 0:
        raise RuntimeError(
            f"invalid declared channel count for {stream_name}: "
            f"{channel_count!r}"
        )
    return channel_count


def _declared_channel_labels(stream, stream_name, channel_count):
    """Read an exact, unique channel-label list from the stream description."""

    description = stream.get("info", {}).get("desc")
    channel_groups = _metadata_values_for_keys(description, {"channels"})
    if len(channel_groups) != 1:
        raise RuntimeError(
            f"{stream_name} must declare exactly one channel-label group; "
            f"found {len(channel_groups)}"
        )
    group = _unwrap_singleton(channel_groups[0])
    if not isinstance(group, dict):
        raise RuntimeError(f"invalid channel-label metadata for {stream_name}")
    channel_key = next(
        (key for key in group if str(key).lower() == "channel"),
        None,
    )
    if channel_key is None:
        raise RuntimeError(f"no channel labels declared for {stream_name}")
    entries = group[channel_key]
    if not isinstance(entries, (list, tuple)):
        entries = [entries]

    labels = []
    for index, entry in enumerate(entries):
        label_values = _metadata_values_for_keys(entry, {"label"})
        if len(label_values) != 1:
            raise RuntimeError(
                f"channel {index} of {stream_name} must declare exactly one label"
            )
        label = str(_unwrap_singleton(label_values[0])).strip()
        if not label:
            raise RuntimeError(f"channel {index} of {stream_name} has no label")
        labels.append(label)

    if len(labels) != channel_count:
        raise RuntimeError(
            f"channel-label mismatch for {stream_name}: metadata declares "
            f"{channel_count} channels but provides {len(labels)} labels"
        )
    if len(set(labels)) != len(labels):
        raise RuntimeError(f"duplicate channel labels in {stream_name}")
    return labels


def _validated_lsl_timestamps(timestamps, sample_count, stream_name):
    """Validate one finite, strictly increasing LSL timestamp per sample."""

    try:
        values = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"timestamps are not numeric for {stream_name}"
        ) from error
    if values.ndim != 1:
        raise RuntimeError(
            f"timestamps for {stream_name} must be one-dimensional; "
            f"got shape {values.shape!r}"
        )
    if len(values) != sample_count:
        raise RuntimeError(
            f"timestamp/sample mismatch for {stream_name}: {len(values)} "
            f"timestamps for {sample_count} samples"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"timestamps are not finite for {stream_name}")
    if len(values) > 1 and np.any(np.diff(values) <= 0):
        raise RuntimeError(
            f"timestamps are not strictly increasing for {stream_name}"
        )
    return values


def _validated_numeric_stream(
    stream,
    stream_name,
    *,
    optional_nan_labels=(),
):
    """Validate a labelled rectangular numeric stream and its LSL timeline."""

    channel_count = _declared_channel_count(stream, stream_name)
    labels = _declared_channel_labels(stream, stream_name, channel_count)
    raw_data = stream.get("time_series")
    if raw_data is None or len(raw_data) == 0:
        raise RuntimeError(f"no samples found in stream: {stream_name}")
    try:
        values = np.asarray(raw_data, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"samples are not a rectangular numeric array for {stream_name}"
        ) from error
    raw_timestamps = stream.get("time_stamps")
    timestamp_count = len(raw_timestamps) if raw_timestamps is not None else 0
    if (
        values.ndim == 1
        and timestamp_count == 1
        and len(values) == channel_count
    ):
        values = values.reshape(1, channel_count)
    if values.ndim != 2:
        raise RuntimeError(
            f"samples for {stream_name} must be a sample-by-channel matrix; "
            f"got shape {values.shape!r}"
        )
    if values.shape[1] != channel_count:
        raise RuntimeError(
            f"channel mismatch for {stream_name}: metadata declares "
            f"{channel_count}, data has {values.shape[1]}"
        )

    optional = set(optional_nan_labels)
    unknown_optional = optional - set(labels)
    if unknown_optional:
        optional = optional & set(labels)
    for index, label in enumerate(labels):
        column = values[:, index]
        if label in optional:
            if np.any(np.isinf(column)):
                raise RuntimeError(
                    f"optional channel {label!r} contains infinity in {stream_name}"
                )
        elif not np.all(np.isfinite(column)):
            raise RuntimeError(
                f"channel {label!r} contains non-finite samples in {stream_name}"
            )

    timestamps = _validated_lsl_timestamps(
        raw_timestamps,
        len(values),
        stream_name,
    )
    return values, timestamps, labels


def _write_dataframe_atomic(dataframe, output_file):
    """Write one CSV without publishing a partial file on failure."""

    output_file = os.fspath(output_file)
    directory = os.path.dirname(output_file)
    basename = os.path.basename(output_file)
    partial_file = os.path.join(directory, f".{basename}.partial")
    try:
        dataframe.to_csv(partial_file, index=False)
        os.replace(partial_file, output_file)
    except Exception:
        try:
            os.remove(partial_file)
        except FileNotFoundError:
            pass
        raise


def _remove_output_path(path):
    """Remove one managed file or directory during rollback."""

    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _publish_staged_outputs(staging_dir, output_dir, relative_paths):
    """Publish a group and roll back every member if any rename fails."""

    os.makedirs(output_dir, exist_ok=True)
    desired = list(relative_paths)
    existing_names = {
        entry.name.casefold(): entry.name
        for entry in os.scandir(output_dir)
        if entry.name != os.path.basename(staging_dir)
    }
    conflicts = [
        relative_path
        for relative_path in desired
        if relative_path.casefold() in existing_names
    ]
    if conflicts:
        raise RuntimeError(
            "managed output already exists: " + ", ".join(sorted(conflicts))
        )

    published = []
    try:
        for relative_path in desired:
            source = os.path.join(staging_dir, relative_path)
            destination = os.path.join(output_dir, relative_path)
            os.replace(source, destination)
            published.append(destination)
    except Exception:
        for destination in reversed(published):
            _remove_output_path(destination)
        raise


def _metadata_values_for_keys(payload, keys):
    """Recursively collect values for metadata keys from pyxdf structures."""
    values = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in keys:
                values.append(_unwrap_singleton(value))
            values.extend(_metadata_values_for_keys(value, keys))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            values.extend(_metadata_values_for_keys(value, keys))
    elif isinstance(payload, np.ndarray) and payload.dtype == object:
        for value in payload.reshape(-1):
            values.extend(_metadata_values_for_keys(value, keys))
    return values


def _validated_depth_scale(value, source):
    """Validate a metres-per-device-unit depth scale."""
    value = _unwrap_singleton(value)
    try:
        scale = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"invalid depth scale from {source}: expected a number, got {value!r}"
        ) from error
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            f"invalid depth scale from {source}: expected a finite positive "
            f"metres-per-unit value, got {value!r}"
        )
    return scale


def _one_consistent_depth_scale(values, source):
    """Return one validated scale, rejecting conflicting metadata."""
    scales = [_validated_depth_scale(value, source) for value in values]
    if not scales:
        return None
    first = scales[0]
    if any(not np.isclose(first, scale, rtol=1e-9, atol=0.0) for scale in scales[1:]):
        raise RuntimeError(
            f"conflicting depth scales in {source}: "
            + ", ".join(f"{scale:.17g}" for scale in scales)
        )
    return first


def _decode_metadata_sample(sample):
    """Decode one JSON metadata sample, returning an empty mapping if unrelated."""
    sample = _unwrap_singleton(sample)
    if isinstance(sample, bytes):
        sample = sample.decode("utf-8")
    if isinstance(sample, str):
        try:
            sample = json.loads(sample)
        except json.JSONDecodeError:
            return {}
    return sample if isinstance(sample, dict) else {}


def _stream_family(name):
    """Return the device-family portion used to pair depth and metadata streams."""
    parts = [part for part in re.split(r"[^a-z0-9]+", name.lower()) if part]
    ignored = {"depth", "metadata", "deviceinfo", "device", "info", "color"}
    return "_".join(part for part in parts if part not in ignored)


def _resolve_depth_scale(depth_stream, all_streams=None):
    """Resolve the recorded hardware scale for one raw depth stream.

    Numeric metadata embedded in the depth stream is preferred. Older NaturalLab
    recordings stored the same value in a paired ``DeviceInfo`` stream, which is
    used only when it can be associated unambiguously.
    """
    stream_name = _stream_name(depth_stream)
    scale_keys = {"depth_scale_m_per_unit", "depth_scale"}
    embedded_values = _metadata_values_for_keys(
        depth_stream.get("info", {}),
        scale_keys,
    )
    embedded_scale = _one_consistent_depth_scale(
        embedded_values,
        f"{stream_name} stream metadata",
    )

    available_streams = list(all_streams or [])
    depth_stream_count = sum(
        _stream_type(candidate).lower() in {"depth", "depthdata"}
        for candidate in available_streams
    )
    metadata_records = []
    for candidate in available_streams:
        candidate_name = _stream_name(candidate)
        if (
            _stream_type(candidate).lower() != "deviceinfo"
            and "metadata" not in candidate_name.lower()
        ):
            continue
        values = []
        for sample in candidate.get("time_series", []):
            values.extend(
                _metadata_values_for_keys(
                    _decode_metadata_sample(sample),
                    scale_keys,
                )
            )
        scale = _one_consistent_depth_scale(
            values,
            f"{candidate_name} metadata stream",
        )
        if scale is not None:
            metadata_records.append((candidate_name, scale))

    depth_family = _stream_family(stream_name)
    paired_records = [
        record
        for record in metadata_records
        if depth_family and _stream_family(record[0]) == depth_family
    ]

    if embedded_scale is not None:
        for metadata_name, metadata_scale in paired_records:
            if not np.isclose(
                embedded_scale,
                metadata_scale,
                rtol=1e-9,
                atol=0.0,
            ):
                raise RuntimeError(
                    f"conflicting depth scales for {stream_name}: stream "
                    f"metadata has {embedded_scale:.17g} m/unit but "
                    f"{metadata_name} has {metadata_scale:.17g} m/unit"
                )
        return embedded_scale, "depth stream metadata"

    candidates = paired_records
    if (
        not candidates
        and len(metadata_records) == 1
        and depth_stream_count == 1
    ):
        candidates = metadata_records
    if candidates:
        scale = _one_consistent_depth_scale(
            [record[1] for record in candidates],
            f"metadata associated with {stream_name}",
        )
        sources = ", ".join(record[0] for record in candidates)
        return scale, f"DeviceInfo stream {sources}"

    if metadata_records:
        names = ", ".join(record[0] for record in metadata_records)
        raise RuntimeError(
            f"cannot associate a recorded depth scale with {stream_name}; "
            f"candidate metadata streams: {names}"
        )
    raise RuntimeError(
        f"no recorded depth scale is available for {stream_name}; metric depth "
        "cannot be produced safely. Record depth_scale_m_per_unit in the depth "
        "stream metadata or a paired DeviceInfo stream."
    )


def _decode_lsl_video_frame(frame_data, frame_index, stream_name):
    """Decode one base64 JPEG frame or fail without changing frame indices."""
    try:
        if isinstance(frame_data, np.ndarray):
            if frame_data.size == 0:
                raise ValueError("empty frame sample")
            frame_str = frame_data.reshape(-1)[0]
        elif isinstance(frame_data, list):
            if not frame_data:
                raise ValueError("empty frame sample")
            frame_str = frame_data[0]
        else:
            frame_str = frame_data
        jpeg_data = base64.b64decode(frame_str, validate=True)
        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as error:
        raise RuntimeError(
            f"could not decode frame {frame_index} from {stream_name}: {error}"
        ) from error
    if frame is None:
        raise RuntimeError(
            f"could not decode frame {frame_index} from {stream_name}"
        )
    return frame


def _verify_video_file(path, expected_frames, expected_size, stream_name):
    """Reopen and fully decode staged video before publishing it."""

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"could not reopen staged video for {stream_name}"
        )
    decoded_frames = 0
    expected_width, expected_height = expected_size
    try:
        while True:
            decoded, frame = capture.read()
            if not decoded:
                break
            if frame is None or frame.shape[:2] != (
                expected_height,
                expected_width,
            ):
                actual_shape = None if frame is None else frame.shape[:2]
                raise RuntimeError(
                    f"staged video frame {decoded_frames} for {stream_name} "
                    f"has shape {actual_shape!r}; expected "
                    f"{(expected_height, expected_width)!r}"
                )
            decoded_frames += 1
            if decoded_frames > expected_frames:
                break
    finally:
        capture.release()
    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"staged video verification failed for {stream_name}: decoded "
            f"{decoded_frames} of {expected_frames} expected frames"
        )


def extract_video_stream(stream, output_dir, name=None, output_stem=None):
    """Extract a video stream while preserving 1:1 timestamp row alignment."""
    stream_name = name or stream['info']['name'][0]
    stem = _resolved_output_stem(stream_name, output_stem)
    print(f"Extracting video stream: {stream_name}")

    output_file = os.path.join(output_dir, f"{stem}.mp4")
    partial_file = os.path.join(output_dir, f".{stem}.partial.mp4")
    timestamp_file = os.path.join(output_dir, f"{stem}_timestamps.csv")
    partial_timestamp_file = os.path.join(
        output_dir,
        f".{stem}_timestamps.partial.csv",
    )
    if os.path.lexists(output_file) or os.path.lexists(timestamp_file):
        raise RuntimeError(
            f"video output already exists for {stream_name}; use an empty "
            "extraction directory"
        )
    timestamps = stream['time_stamps']
    frames_data = stream['time_series']

    if frames_data is None or len(frames_data) == 0:
        raise RuntimeError(f"no video frames found in stream: {stream_name}")
    if len(timestamps) != len(frames_data):
        raise RuntimeError(
            f"video/timestamp length mismatch for {stream_name}: "
            f"{len(frames_data)} frames and {len(timestamps)} timestamps"
        )

    timestamp_values = _validated_lsl_timestamps(
        timestamps,
        len(frames_data),
        stream_name,
    )
    frame_intervals = np.diff(timestamp_values)
    avg_interval = np.mean(frame_intervals) if len(frame_intervals) > 0 else 1 / 30
    fps = 1.0 / avg_interval if avg_interval > 0 else 30
    print(f"Estimated frame rate: {fps:.2f} FPS")

    first_frame = _decode_lsl_video_frame(frames_data[0], 0, stream_name)
    height, width = first_frame.shape[:2]
    print(f"Frame dimensions: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(partial_file, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        video_writer.release()
        raise RuntimeError(f"could not create video file for {stream_name}")

    print(f"Processing {len(frames_data)} frames...")
    try:
        for frame_index, frame_data in enumerate(tqdm(frames_data)):
            frame = _decode_lsl_video_frame(
                frame_data,
                frame_index,
                stream_name,
            )
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"frame size changed at frame {frame_index} in "
                    f"{stream_name}"
                )
            video_writer.write(frame)
    except Exception:
        video_writer.release()
        try:
            os.remove(partial_file)
        except FileNotFoundError:
            pass
        raise
    else:
        video_writer.release()

    timestamp_df = pd.DataFrame({
        'frame_index': range(len(timestamp_values)),
        'timestamp': timestamp_values,
        'timestamp_domain': ['lsl'] * len(timestamp_values),
    })
    committed_video = False
    try:
        _verify_video_file(
            partial_file,
            len(frames_data),
            (width, height),
            stream_name,
        )
        timestamp_df.to_csv(partial_timestamp_file, index=False)
        os.replace(partial_file, output_file)
        committed_video = True
        os.replace(partial_timestamp_file, timestamp_file)
    except Exception:
        if committed_video:
            try:
                os.remove(output_file)
            except FileNotFoundError:
                pass
        try:
            os.remove(partial_file)
        except FileNotFoundError:
            pass
        try:
            os.remove(partial_timestamp_file)
        except FileNotFoundError:
            pass
        raise

    print(f"Video saved to: {output_file}")
    print(f"Timestamps saved to: {timestamp_file}")

def extract_audio_stream(stream, output_dir, name=None, output_stem=None):
    """Extract one sample-aligned LSL audio stream to WAV and timestamps."""
    stream_name = name or stream['info']['name'][0]
    stem = _resolved_output_stem(stream_name, output_stem)
    print(f"Extracting audio stream: {stream_name}")

    output_file = os.path.join(output_dir, f"{stem}.wav")
    partial_file = os.path.join(output_dir, f".{stem}.partial.wav")
    timestamp_file = os.path.join(output_dir, f"{stem}_timestamps.csv")
    partial_timestamp_file = os.path.join(
        output_dir,
        f".{stem}_timestamps.partial.csv",
    )
    if os.path.lexists(output_file) or os.path.lexists(timestamp_file):
        raise RuntimeError(
            f"audio output already exists for {stream_name}; use an empty "
            "extraction directory"
        )
    timestamps = np.asarray(stream['time_stamps'], dtype=np.float64)
    audio_data = stream['time_series']

    if audio_data is None or len(audio_data) == 0:
        raise RuntimeError(f"no audio samples found in stream: {stream_name}")

    sample_rate_value = float(stream['info']['nominal_srate'][0])
    if not np.isfinite(sample_rate_value) or sample_rate_value <= 0:
        raise RuntimeError(
            f"invalid nominal sample rate for {stream_name}: {sample_rate_value!r}"
        )
    sample_rate = int(round(sample_rate_value))
    if not np.isclose(sample_rate, sample_rate_value):
        raise RuntimeError(
            f"non-integral nominal sample rate for {stream_name}: "
            f"{sample_rate_value!r}"
        )
    print(f"Sample rate: {sample_rate} Hz")

    channel_count = int(stream['info']['channel_count'][0])
    if channel_count <= 0:
        raise RuntimeError(
            f"invalid channel count for {stream_name}: {channel_count!r}"
        )
    print(f"Channels: {channel_count}")

    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError(
            "audio extraction requires soundfile; install "
            "NaturalLab's acquisition extra"
        ) from error

    try:
        audio_array = np.asarray(audio_data, dtype=np.float32)
        if audio_array.ndim == 1:
            audio_array = audio_array.reshape(-1, 1)
        if audio_array.ndim != 2 or audio_array.shape[1] != channel_count:
            raise RuntimeError(
                f"audio shape {audio_array.shape!r} does not match the declared "
                f"{channel_count} channel(s)"
            )
        if len(timestamps) != len(audio_array):
            raise RuntimeError(
                f"audio timestamp/sample mismatch for {stream_name}: "
                f"{len(timestamps)} timestamps for {len(audio_array)} samples"
            )
        if not np.all(np.isfinite(audio_array)):
            raise RuntimeError(f"audio contains non-finite samples: {stream_name}")
        if not np.all(np.isfinite(timestamps)):
            raise RuntimeError(f"audio contains non-finite timestamps: {stream_name}")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise RuntimeError(
                f"audio timestamps are not strictly increasing: {stream_name}"
            )

        print(f"Audio array shape: {audio_array.shape}")
        sf.write(partial_file, audio_array, sample_rate)
        timestamp_df = pd.DataFrame({
            'sample_index': range(len(timestamps)),
            'timestamp': timestamps,
            'timestamp_domain': ['lsl'] * len(timestamps),
        })
        timestamp_df.to_csv(partial_timestamp_file, index=False)
        os.replace(partial_file, output_file)
        try:
            os.replace(partial_timestamp_file, timestamp_file)
        except Exception:
            try:
                os.remove(output_file)
            except FileNotFoundError:
                pass
            raise
        print(f"Audio saved to: {output_file}")
        print(f"Timestamps saved to: {timestamp_file}")
        return output_file
    except Exception:
        try:
            os.remove(partial_file)
        except FileNotFoundError:
            pass
        try:
            os.remove(partial_timestamp_file)
        except FileNotFoundError:
            pass
        raise

def extract_gaze_stream(stream, output_dir, name=None, output_stem=None):
    """Extract a declared gaze stream without guessing or changing its data."""

    stream_name = name or stream["info"]["name"][0]
    stem = _resolved_output_stem(stream_name, output_stem)
    print(f"Extracting gaze stream: {stream_name}")
    gaze_data, timestamps, labels = _validated_numeric_stream(
        stream,
        stream_name,
    )
    reserved = {"timestamp", "timestamp_domain"}
    if reserved & set(labels):
        raise RuntimeError(
            f"gaze channel labels conflict with extraction metadata: {stream_name}"
        )
    dataframe = pd.DataFrame(gaze_data, columns=labels)
    dataframe["timestamp"] = timestamps
    dataframe["timestamp_domain"] = "lsl"
    output_file = os.path.join(output_dir, f"{stem}.csv")
    _write_dataframe_atomic(dataframe, output_file)
    print(f"Gaze data saved to: {output_file}")
    return output_file

def extract_metadata_stream(stream, output_dir, name=None, output_stem=None):
    """Extract a metadata stream from XDF to JSON"""
    stream_name = name or stream['info']['name'][0]
    stem = _resolved_output_stem(stream_name, output_stem)
    print(f"Extracting metadata stream: {stream_name}")
    
    # Create output JSON file
    output_file = os.path.join(output_dir, f"{stem}.json")
    
    # Extract timestamps and metadata
    timestamps = stream['time_stamps']
    metadata_entries = stream['time_series']
    
    if metadata_entries is None or len(metadata_entries) == 0:
        raise RuntimeError(f"no metadata samples found in stream: {stream_name}")
    timestamp_values = _validated_lsl_timestamps(
        timestamps,
        len(metadata_entries),
        stream_name,
    )
    
    # Process metadata with progress bar
    metadata_list = []
    for i, entry in enumerate(tqdm(metadata_entries, desc="Processing metadata")):
        if isinstance(entry, np.ndarray):
            if entry.size != 1:
                raise RuntimeError(
                    f"metadata sample {i} in {stream_name} must have one channel"
                )
            entry_data = entry.reshape(-1)[0]
        elif isinstance(entry, list):
            if len(entry) != 1:
                raise RuntimeError(
                    f"metadata sample {i} in {stream_name} must have one channel"
                )
            entry_data = entry[0]
        else:
            entry_data = entry

        if isinstance(entry_data, str):
            try:
                metadata = json.loads(entry_data)
            except json.JSONDecodeError:
                metadata = entry_data
        else:
            metadata = entry_data

        metadata_list.append(
            {
                "timestamp": timestamp_values[i],
                "timestamp_domain": "lsl",
                "metadata": metadata,
            }
        )
    
    # Save to JSON file
    partial_file = os.path.join(output_dir, f".{stem}.partial.json")
    try:
        with open(partial_file, "w", encoding="utf-8") as file_handle:
            json.dump(metadata_list, file_handle, indent=2)
            file_handle.write("\n")
        os.replace(partial_file, output_file)
    except Exception:
        try:
            os.remove(partial_file)
        except FileNotFoundError:
            pass
        raise
    
    print(f"Metadata saved to: {output_file}")

def _decode_lsl_depth_frame(frame_data, frame_index, stream_name):
    """Decode one raw uint16 depth PNG or fail without skipping the frame."""

    try:
        if isinstance(frame_data, np.ndarray):
            if frame_data.size == 0:
                raise ValueError("empty frame sample")
            frame_str = frame_data.reshape(-1)[0]
        elif isinstance(frame_data, list):
            if not frame_data:
                raise ValueError("empty frame sample")
            frame_str = frame_data[0]
        else:
            frame_str = frame_data
        png_data = base64.b64decode(frame_str, validate=True)
        raw_depth = cv2.imdecode(
            np.frombuffer(png_data, np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    except Exception as error:
        raise RuntimeError(
            f"could not decode depth frame {frame_index} from "
            f"{stream_name}: {error}"
        ) from error
    if raw_depth is None or raw_depth.size == 0:
        raise RuntimeError(
            f"could not decode depth frame {frame_index} from {stream_name}"
        )
    if raw_depth.ndim != 2 or raw_depth.dtype != np.uint16:
        raise RuntimeError(
            f"depth frame {frame_index} from {stream_name} is "
            f"{raw_depth.dtype} with shape {raw_depth.shape!r}; expected one "
            "uint16 channel of raw device values"
        )
    return raw_depth


def extract_depth_stream(
    stream,
    output_dir,
    name=None,
    output_stem=None,
    save_interval=30,
    include_csv=False,
    depth_scale_m_per_unit=None,
    depth_scale_source=None,
):
    """Transactionally extract raw depth and verified metric derivatives."""

    stream_name = name or stream["info"]["name"][0]
    stem = _resolved_output_stem(stream_name, output_stem)
    print(f"Extracting depth stream: {stream_name}")

    timestamps = stream["time_stamps"]
    frames_data = stream["time_series"]

    if frames_data is None or len(frames_data) == 0:
        raise RuntimeError(f"no depth frames found in stream: {stream_name}")
    if save_interval <= 0:
        raise ValueError("depth save interval must be a positive integer")
    timestamp_values = _validated_lsl_timestamps(
        timestamps,
        len(frames_data),
        stream_name,
    )
    frame_intervals = np.diff(timestamp_values)

    if depth_scale_m_per_unit is None:
        depth_scale, resolved_source = _resolve_depth_scale(stream, [stream])
        depth_scale_source = depth_scale_source or resolved_source
    else:
        depth_scale = _validated_depth_scale(
            depth_scale_m_per_unit,
            depth_scale_source or f"{stream_name} extraction metadata",
        )
        depth_scale_source = depth_scale_source or "extraction metadata"
    print(
        "Depth scale: "
        f"{depth_scale!r} metres per raw device unit "
        f"(source: {depth_scale_source})"
    )

    avg_interval = np.mean(frame_intervals) if len(frame_intervals) > 0 else 1/30
    fps = 1.0 / avg_interval if avg_interval > 0 else 30
    print(f"Estimated frame rate: {fps:.2f} FPS")

    first_raw_depth = _decode_lsl_depth_frame(
        frames_data[0],
        0,
        stream_name,
    )

    depth_min_global = float('inf')
    depth_max_global = 0
    valid_depths = []
    valid_mask = first_raw_depth > 0
    if np.any(valid_mask):
        depth_min_global = np.min(first_raw_depth[valid_mask])
        depth_max_global = np.max(first_raw_depth[valid_mask])
        sample_size = min(10000, np.count_nonzero(valid_mask))
        valid_indices = np.where(valid_mask.flatten())[0]
        sampled_indices = np.random.choice(
            valid_indices,
            sample_size,
            replace=False,
        )
        valid_depths.extend(first_raw_depth.flatten()[sampled_indices])

    if valid_depths:
        valid_depths = np.array(valid_depths)
        p_low = np.percentile(valid_depths, 1)
        p_high = np.percentile(valid_depths, 99)
        range_expand = (p_high - p_low) * 0.1
        vis_min = max(0, p_low - range_expand)
        vis_max = min(65535, p_high + range_expand)
        print(f"Using depth range for visualization: {vis_min:.1f}-{vis_max:.1f}")
        print(
            "This corresponds to approximately "
            f"{vis_min * depth_scale:.3f}m - {vis_max * depth_scale:.3f}m"
        )
    else:
        vis_min = depth_min_global if depth_min_global != float('inf') else 0
        vis_max = depth_max_global if depth_max_global != 0 else 10000
        print(f"Fallback depth range: {vis_min}-{vis_max}")

    height, width = first_raw_depth.shape[:2]
    print(f"Frame dimensions: {width}x{height}")

    depth_dir_name = f"{stem}_depth"
    visualization_name = f"{stem}_visualization.mp4"
    timestamp_name = f"{stem}_timestamps.csv"
    metadata_name = f"{stem}_depth_metadata.json"
    managed_outputs = (
        depth_dir_name,
        visualization_name,
        timestamp_name,
        metadata_name,
    )
    output_dir = os.path.abspath(os.fspath(output_dir))
    if os.path.lexists(output_dir):
        if os.path.islink(output_dir) or not os.path.isdir(output_dir):
            raise RuntimeError(f"depth output path is not a directory: {output_dir}")
    else:
        os.makedirs(output_dir)
    staging_dir = tempfile.mkdtemp(
        prefix=f".{stem}.depth-staging-",
        dir=output_dir,
    )
    staged_depth_dir = os.path.join(staging_dir, depth_dir_name)
    staged_visualization = os.path.join(staging_dir, visualization_name)
    staged_timestamps = os.path.join(staging_dir, timestamp_name)
    staged_metadata = os.path.join(staging_dir, metadata_name)
    os.makedirs(staged_depth_dir)

    print(f"Processing {len(frames_data)} frames...")
    print(f"Saving raw depth PNG every {save_interval} frames")

    frame_counter = 0
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            staged_visualization,
            fourcc,
            fps,
            (width, height),
        )
        if not video_writer.isOpened():
            video_writer.release()
            raise RuntimeError(
                f"could not create depth visualization for {stream_name}"
            )
        try:
            for i, frame_data in enumerate(tqdm(frames_data)):
                raw_depth = _decode_lsl_depth_frame(
                    frame_data,
                    i,
                    stream_name,
                )
                if raw_depth.shape != first_raw_depth.shape:
                    raise RuntimeError(
                        f"depth frame size changed at frame {i} in "
                        f"{stream_name}: {raw_depth.shape!r} != "
                        f"{first_raw_depth.shape!r}"
                    )

                valid_mask = raw_depth > 0
                color_frame = np.zeros(
                    (raw_depth.shape[0], raw_depth.shape[1], 3),
                    dtype=np.uint8,
                )
                if np.any(valid_mask):
                    normalized = np.zeros_like(raw_depth, dtype=np.uint8)
                    visualization_span = max(float(vis_max - vis_min), 1.0)
                    normalized[valid_mask] = np.clip(
                        (
                            (raw_depth[valid_mask] - vis_min)
                            / visualization_span
                            * 255
                        ),
                        0,
                        255,
                    ).astype(np.uint8)
                    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                    color_frame[valid_mask] = colored[valid_mask]
                    cv2.putText(
                        color_frame,
                        (
                            f"Range: {vis_min * depth_scale:.2f}m - "
                            f"{vis_max * depth_scale:.2f}m"
                        ),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                if i % save_interval == 0:
                    depth_file = os.path.join(
                        staged_depth_dir,
                        f"depth_{i:06d}.png",
                    )
                    if not cv2.imwrite(depth_file, raw_depth):
                        raise RuntimeError(
                            f"could not write raw depth frame {i} for "
                            f"{stream_name}"
                        )
                    frame_counter += 1
                    if include_csv:
                        distance_map = raw_depth.astype(np.float32) * depth_scale
                        distance_file = os.path.join(
                            staged_depth_dir,
                            f"distance_{i:06d}.csv",
                        )
                        np.savetxt(distance_file, distance_map, delimiter=',')

                video_writer.write(color_frame)
        finally:
            video_writer.release()

        _verify_video_file(
            staged_visualization,
            len(frames_data),
            (width, height),
            stream_name,
        )
        timestamp_df = pd.DataFrame({
            'frame_index': range(len(timestamp_values)),
            'timestamp': timestamp_values,
            'timestamp_domain': ['lsl'] * len(timestamp_values),
        })
        timestamp_df.to_csv(staged_timestamps, index=False)
        depth_metadata = {
            "stream_name": stream_name,
            "output_stem": stem,
            "raw_encoding": f"{first_raw_depth.dtype} PNG",
            "raw_value_unit": "device_depth_unit",
            "depth_scale_m_per_unit": depth_scale,
            "depth_scale_source": depth_scale_source,
            "metric_distance_unit": "metre",
            "distance_csv_unit": "metre" if include_csv else None,
        }
        with open(staged_metadata, "w", encoding="utf-8") as file_handle:
            json.dump(depth_metadata, file_handle, indent=2)
            file_handle.write("\n")
        _publish_staged_outputs(staging_dir, output_dir, managed_outputs)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    else:
        shutil.rmtree(staging_dir, ignore_errors=True)

    depth_dir = os.path.join(output_dir, depth_dir_name)
    output_file = os.path.join(output_dir, visualization_name)
    timestamp_file = os.path.join(output_dir, timestamp_name)
    depth_metadata_file = os.path.join(output_dir, metadata_name)
    print(f"Depth visualization saved to: {output_file}")
    print(f"Raw depth samples ({frame_counter} frames) saved to: {depth_dir}")
    print(f"Timestamps saved to: {timestamp_file}")
    print(f"Depth metadata saved to: {depth_metadata_file}")
    print(
        "Raw depth PNG values are device units; multiply by "
        f"{depth_scale!r} metres per unit for metric distances"
    )
    return depth_metadata


def extract_generic_stream(stream, output_dir, name=None, output_stem=None):
    """Extract an explicitly labelled unknown stream without type guessing."""

    stream_name = name or stream['info']['name'][0]
    stem = _resolved_output_stem(stream_name, output_stem)
    stream_type = stream['info']['type'][0]
    print(f"Extracting generic stream: {stream_name} (type: {stream_type})")

    channel_count = _declared_channel_count(stream, stream_name)
    labels = _declared_channel_labels(stream, stream_name, channel_count)
    if {"timestamp", "timestamp_domain"} & set(labels):
        raise RuntimeError(
            f"generic channel labels conflict with extraction metadata: {stream_name}"
        )
    data_series = stream.get("time_series")
    if data_series is None or len(data_series) == 0:
        raise RuntimeError(f"no samples found in stream: {stream_name}")
    try:
        values = np.asarray(data_series)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"samples are not rectangular for generic stream {stream_name}"
        ) from error
    raw_timestamps = stream.get("time_stamps")
    timestamp_count = len(raw_timestamps) if raw_timestamps is not None else 0
    if values.ndim == 1:
        if channel_count == 1 and len(values) == timestamp_count:
            values = values.reshape(-1, 1)
        elif timestamp_count == 1 and len(values) == channel_count:
            values = values.reshape(1, channel_count)
    if values.ndim != 2 or values.shape[1] != channel_count:
        raise RuntimeError(
            f"generic stream {stream_name} must be a sample-by-channel matrix "
            f"with {channel_count} channels; got {values.shape!r}"
        )
    if values.dtype.kind in "iufc" and not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"generic numeric stream contains non-finite values: {stream_name}"
        )
    if values.dtype.kind == "O":
        for value in values.reshape(-1):
            if value is None or isinstance(value, (list, tuple, dict, np.ndarray)):
                raise RuntimeError(
                    f"generic stream contains a non-scalar value: {stream_name}"
                )
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise RuntimeError(
                    f"generic stream contains a non-finite value: {stream_name}"
                )

    timestamps = _validated_lsl_timestamps(
        raw_timestamps,
        len(values),
        stream_name,
    )
    dataframe = pd.DataFrame(values, columns=labels)
    dataframe["timestamp"] = timestamps
    dataframe["timestamp_domain"] = "lsl"
    output_file = os.path.join(output_dir, f"{stem}.csv")
    _write_dataframe_atomic(dataframe, output_file)
    print(f"Data saved to: {output_file}")
    return output_file
            
def extract_imu_stream(
    stream,
    output_dir,
    name=None,
    output_filename=None,
    output_stem=None,
):
    """Extract one validated IMU stream without changing its LSL clock."""
    stream_name = name or stream['info']['name'][0]
    stem = _resolved_output_stem(stream_name, output_stem)
    print(f"Extracting IMU stream: {stream_name}")

    if output_filename is not None:
        if os.path.basename(output_filename) != output_filename:
            raise ValueError("IMU output filename must not contain a directory")
        requested_stem, requested_extension = os.path.splitext(output_filename)
        if requested_extension.lower() != ".csv":
            raise ValueError("IMU output filename must end in .csv")
        stem = _resolved_output_stem(stream_name, requested_stem)
    output_filename = f"{stem}.csv"
    output_file = os.path.join(output_dir, output_filename)
    partial_file = os.path.join(output_dir, f".{output_filename}.partial")

    try:
        channel_count = int(stream['info']['channel_count'][0])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"invalid channel count for IMU stream {stream_name}"
        ) from error
    if channel_count <= 0:
        raise RuntimeError(
            f"invalid channel count for IMU stream {stream_name}: "
            f"{channel_count!r}"
        )

    raw_imu_data = stream.get('time_series')
    if raw_imu_data is None or len(raw_imu_data) == 0:
        raise RuntimeError(f"no IMU samples found in stream: {stream_name}")
    try:
        imu_data = np.asarray(raw_imu_data, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"IMU samples are not a rectangular numeric array: {stream_name}"
        ) from error

    try:
        timestamps = np.asarray(stream['time_stamps'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"IMU timestamps are not a numeric array: {stream_name}"
        ) from error

    # A single recorded sample may be represented as a flat channel vector.
    if imu_data.ndim == 1 and len(timestamps) == 1 and len(imu_data) == channel_count:
        imu_data = imu_data.reshape(1, channel_count)
    if imu_data.ndim != 2:
        raise RuntimeError(
            f"IMU data for {stream_name} must be a sample-by-channel matrix; "
            f"got shape {imu_data.shape!r}"
        )
    if imu_data.shape[1] != channel_count:
        raise RuntimeError(
            f"IMU channel mismatch for {stream_name}: metadata declares "
            f"{channel_count}, data has {imu_data.shape[1]}"
        )
    if timestamps.ndim != 1:
        raise RuntimeError(
            f"IMU timestamps for {stream_name} must be one-dimensional; "
            f"got shape {timestamps.shape!r}"
        )
    if len(timestamps) != len(imu_data):
        raise RuntimeError(
            f"IMU timestamp/sample mismatch for {stream_name}: "
            f"{len(timestamps)} timestamps for {len(imu_data)} samples"
        )
    if not np.all(np.isfinite(imu_data)):
        raise RuntimeError(f"IMU contains non-finite samples: {stream_name}")
    if not np.all(np.isfinite(timestamps)):
        raise RuntimeError(f"IMU contains non-finite timestamps: {stream_name}")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise RuntimeError(
            f"IMU timestamps are not strictly increasing: {stream_name}"
        )

    imu_columns = [
        "gyro_x [deg/s]",
        "gyro_y [deg/s]",
        "gyro_z [deg/s]",
        "accel_x [g]",
        "accel_y [g]",
        "accel_z [g]",
        "roll [deg]",
        "pitch [deg]",
        "yaw [deg]",
        "quaternion_w",
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
    ]
    column_names = imu_columns[:channel_count]
    column_names.extend(
        f"extra_{index}" for index in range(len(imu_columns), channel_count)
    )

    sensor_data = pd.DataFrame(imu_data, columns=column_names)
    metadata = pd.DataFrame(
        {
            "section_id": np.ones(len(timestamps), dtype=np.int64),
            "recording_id": np.ones(len(timestamps), dtype=np.int64),
            "timestamp": timestamps,
            "timestamp_domain": ["lsl"] * len(timestamps),
        }
    )
    final_df = pd.concat([metadata, sensor_data], axis=1)

    try:
        final_df.to_csv(partial_file, index=False)
        os.replace(partial_file, output_file)
    except Exception:
        try:
            os.remove(partial_file)
        except FileNotFoundError:
            pass
        raise

    print(f"IMU data saved to: {output_file}")
    return output_file

def extract_fixations_stream(stream, output_dir, name=None, output_stem=None):
    """Extract declared fixation events without padding or timestamp rewriting."""

    stream_name = name or stream["info"]["name"][0]
    stem = _resolved_output_stem(stream_name, output_stem)
    print(f"Extracting fixations stream: {stream_name}")
    optional_labels = {
        "fixation_x_px",
        "fixation_y_px",
        "mean_gaze_x",
        "mean_gaze_y",
        "azimuth_deg",
        "elevation_deg",
    }
    values, timestamps, labels = _validated_numeric_stream(
        stream,
        stream_name,
        optional_nan_labels=optional_labels,
    )
    if {"timestamp", "timestamp_domain"} & set(labels):
        raise RuntimeError(
            f"fixation labels conflict with extraction metadata: {stream_name}"
        )
    dataframe = pd.DataFrame(values, columns=labels)
    dataframe["timestamp"] = timestamps
    dataframe["timestamp_domain"] = "lsl"
    output_file = os.path.join(output_dir, f"{stem}.csv")
    _write_dataframe_atomic(dataframe, output_file)
    print(f"Fixations data saved to: {output_file}")
    return output_file

def extract_saccades_stream(stream, output_dir, name=None, output_stem=None):
    """Extract declared saccades without padding or timestamp rewriting."""

    stream_name = name or stream["info"]["name"][0]
    stem = _resolved_output_stem(stream_name, output_stem)
    print(f"Extracting saccades stream: {stream_name}")
    optional_labels = {
        "amplitude_deg",
        "amplitude_px",
        "amplitude_angle_deg",
        "amplitude_pixels",
        "mean_velocity_px_s",
        "peak_velocity_px_s",
        "mean_velocity",
        "max_velocity",
    }
    values, timestamps, labels = _validated_numeric_stream(
        stream,
        stream_name,
        optional_nan_labels=optional_labels,
    )
    if {"timestamp", "timestamp_domain"} & set(labels):
        raise RuntimeError(
            f"saccade labels conflict with extraction metadata: {stream_name}"
        )
    dataframe = pd.DataFrame(values, columns=labels)
    dataframe["timestamp"] = timestamps
    dataframe["timestamp_domain"] = "lsl"
    output_file = os.path.join(output_dir, f"{stem}.csv")
    _write_dataframe_atomic(dataframe, output_file)
    print(f"Saccades data saved to: {output_file}")
    return output_file

def extract_streams(
    xdf_file,
    output_dir,
    keep_raw_depth=True,
    depth_interval=30,
    include_csv=False,
):
    """Extract an XDF file transactionally into a new or empty directory."""

    if pyxdf is None:
        raise RuntimeError(
            "pyxdf is required for XDF extraction; install "
            "naturallab[acquisition]"
        )
    if depth_interval <= 0:
        raise ValueError("depth_interval must be a positive integer")
    print(f"Loading XDF file: {xdf_file}")
    print(
        "Raw depth data will be "
        f"{'kept' if keep_raw_depth else 'deleted'} after processing"
    )
    print(f"Saving raw depth PNG every {depth_interval} frames")

    staging_dir = None
    final_output_dir = os.path.abspath(os.fspath(output_dir))
    try:
        streams, _fileheader = pyxdf.load_xdf(xdf_file)
        if not streams:
            raise RuntimeError("XDF file contains no streams")
        output_stems = _plan_stream_output_stems(streams)
        final_output_dir, staging_dir = _prepare_extraction_staging_dir(
            final_output_dir
        )

        print(f"XDF file loaded. Found {len(streams)} streams:")
        for index, stream in enumerate(streams):
            name = _stream_name(stream)
            stream_type = _stream_type(stream)
            channel_count = _declared_channel_count(stream, name)
            sample_count = len(stream.get("time_series", []))
            print(
                f"  {index + 1}. {name} (Type: {stream_type}, "
                f"Channels: {channel_count}, Samples: {sample_count})"
            )

        depth_raw_dirs = []
        found_imu = False
        found_fixations = False
        found_saccades = False
        extraction_errors = []
        imu_outputs = []
        depth_outputs = []

        for stream in streams:
            name = _stream_name(stream)
            stem = output_stems[id(stream)]
            try:
                normalized_type = _stream_type(stream).strip().lower()
                if _is_declared_imu_stream(stream):
                    found_imu = True
                    output_path = extract_imu_stream(
                        stream,
                        staging_dir,
                        output_stem=stem,
                    )
                    imu_outputs.append((name, os.path.basename(output_path)))
                elif normalized_type == "fixations":
                    found_fixations = True
                    extract_fixations_stream(
                        stream,
                        staging_dir,
                        output_stem=stem,
                    )
                elif normalized_type == "saccades":
                    found_saccades = True
                    extract_saccades_stream(
                        stream,
                        staging_dir,
                        output_stem=stem,
                    )
                elif normalized_type == "videostream":
                    extract_video_stream(
                        stream,
                        staging_dir,
                        output_stem=stem,
                    )
                elif normalized_type in {"depth", "depthdata"}:
                    depth_dir = os.path.join(staging_dir, f"{stem}_depth")
                    if not keep_raw_depth:
                        depth_raw_dirs.append((name, depth_dir))
                    depth_scale, scale_source = _resolve_depth_scale(
                        stream,
                        streams,
                    )
                    depth_outputs.append(
                        extract_depth_stream(
                            stream,
                            staging_dir,
                            output_stem=stem,
                            save_interval=depth_interval,
                            include_csv=include_csv,
                            depth_scale_m_per_unit=depth_scale,
                            depth_scale_source=scale_source,
                        )
                    )
                elif normalized_type == "audio":
                    extract_audio_stream(
                        stream,
                        staging_dir,
                        output_stem=stem,
                    )
                elif normalized_type == "gaze":
                    extract_gaze_stream(
                        stream,
                        staging_dir,
                        output_stem=stem,
                    )
                elif normalized_type == "deviceinfo":
                    extract_metadata_stream(
                        stream,
                        staging_dir,
                        output_stem=stem,
                    )
                else:
                    extract_generic_stream(
                        stream,
                        staging_dir,
                        output_stem=stem,
                    )
            except Exception as stream_error:
                print(f"Error extracting stream {name}: {stream_error}")
                extraction_errors.append(f"{name}: {stream_error}")

        if not keep_raw_depth:
            for depth_name, raw_dir in depth_raw_dirs:
                try:
                    if os.path.exists(raw_dir):
                        print(f"Removing raw depth data folder: {raw_dir}")
                        shutil.rmtree(raw_dir)
                except Exception as cleanup_error:
                    extraction_errors.append(
                        f"raw-depth cleanup for {depth_name}: "
                        f"{cleanup_error}"
                    )
        else:
            print("Keeping all raw depth data for future measurement purposes")

        print("\nSpecialized streams extraction summary:")
        if imu_outputs:
            imu_summary = ", ".join(
                f"{stream_name} -> {filename}"
                for stream_name, filename in imu_outputs
            )
            print(
                f"- IMU data: {len(imu_outputs)} stream(s) extracted: "
                f"{imu_summary}"
            )
        elif found_imu:
            print("- IMU data: Found, but no CSV was produced")
        else:
            print("- IMU data: Not found")
        if depth_outputs:
            depth_summary = ", ".join(
                f"{result['stream_name']} "
                f"({result['depth_scale_m_per_unit']!r} m/unit)"
                for result in depth_outputs
            )
            print(
                f"- Depth data: {len(depth_outputs)} stream(s) extracted: "
                f"{depth_summary}"
            )
        else:
            print("- Depth data: Not found or not safely extractable")
        print(
            "- Fixations data: "
            f"{'Extracted' if found_fixations else 'Not found'}"
        )
        print(
            "- Saccades data: "
            f"{'Extracted' if found_saccades else 'Not found'}"
        )

        if extraction_errors:
            summary = "; ".join(extraction_errors[:5])
            if len(extraction_errors) > 5:
                summary += f"; and {len(extraction_errors) - 5} more"
            raise RuntimeError(
                f"XDF extraction was incomplete ({len(extraction_errors)} "
                f"error(s)): {summary}"
            )

        _publish_extraction_staging_dir(staging_dir, final_output_dir)
        staging_dir = None
        print(f"\nAll streams extracted to: {final_output_dir}")
        return final_output_dir
    except Exception as error:
        print(f"Error extracting XDF file: {error}")
        raise
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="XDF Extractor")
    parser.add_argument("--file", type=str, required=True, help="Path to XDF file")
    parser.add_argument("--outdir", type=str, default="extracted_data", help="Output directory (default: extracted_data)")
    parser.add_argument("--no-raw-depth", action="store_true", 
                        help="Delete raw depth data after creating MP4 (not recommended for measurements)")
    parser.add_argument("--depth-interval", type=int, default=1,
                        help="Save raw depth PNG every N frames (default: 1)")
    parser.add_argument("--include-csv", action="store_true", 
                        help="Include CSV distance maps (increases disk usage)")
    args = parser.parse_args()
    
    # Check if XDF file exists
    if not os.path.exists(args.file):
        print(f"Error: XDF file not found: {args.file}")
        return 1
    
    # Start extraction
    start_time = time.time()
    try:
        extract_streams(
            args.file,
            args.outdir,
            keep_raw_depth=not args.no_raw_depth,
            depth_interval=args.depth_interval,
            include_csv=args.include_csv,
        )
    except Exception as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 1
    end_time = time.time()
    
    print(f"\nExtraction completed in {end_time - start_time:.2f} seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
