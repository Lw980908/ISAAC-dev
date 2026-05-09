#!/usr/bin/env python3
import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _fmt_float(x: Any, ndigits: int = 3) -> str:
    if x is None:
        return "null"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def _parse_labels(s: str) -> Set[int]:
    out: Set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _extract_json_from_echo_line(line: str) -> Optional[str]:
    m = re.match(r"^\s*data:\s*(.*)\s*$", line)
    if not m:
        return None
    payload = m.group(1).strip()
    if not payload:
        return None
    if len(payload) >= 2 and payload[0] == payload[-1] and payload[0] in ("'", '"'):
        payload = payload[1:-1]
    return payload


def _print_payload(
    payload: Dict[str, Any],
    *,
    min_score: float,
    min_points_box: int,
    min_points_center: int,
    require_distance: bool,
    only_roi: Optional[str],
    labels: Optional[Set[int]],
    show_center: bool,
    pretty: bool,
    show_all_keys: bool,
) -> None:
    if show_all_keys:
        keys = sorted(payload.keys())
        print("payload_keys:", keys)

    camera_id = payload.get("camera_id")
    distance_frame = payload.get("distance_frame")
    calib_valid = payload.get("calib_valid")
    cloud_age_ms = payload.get("cloud_age_ms")
    cloud_points = payload.get("cloud_points")
    stamp_ns = payload.get("stamp_ns")
    cloud_stamp_ns = payload.get("cloud_stamp_ns")

    hdr = (
        f"[camera_id={camera_id}]"
        + (f" frame={distance_frame}" if distance_frame is not None else "")
        + f" calib_valid={calib_valid}"
        + f" cloud_age_ms={_fmt_float(cloud_age_ms, 1)}"
        + f" cloud_points={cloud_points}"
        + f" stamp_ns={stamp_ns}"
        + f" cloud_stamp_ns={cloud_stamp_ns}"
    )
    print("\n" + hdr)

    objects: List[Dict[str, Any]] = payload.get("objects") or []
    if not objects:
        print("  (no objects)")
        return

    for i, o in enumerate(objects):
        label = o.get("label")
        if labels is not None:
            try:
                if int(label) not in labels:
                    continue
            except Exception:
                continue

        score = o.get("score")
        try:
            if score is not None and float(score) < min_score:
                continue
        except Exception:
            pass

        roi = o.get("roi")
        if only_roi is not None and roi != only_roi:
            continue

        pts_box = o.get("points_in_box", 0)
        pts_center = o.get("points_in_center", 0)
        try:
            if int(pts_box) < min_points_box:
                continue
        except Exception:
            pass
        try:
            if int(pts_center) < min_points_center:
                continue
        except Exception:
            pass

        range_m = o.get("range_m")
        depth_m = o.get("depth_m")
        height_m = o.get("height_m")
        if require_distance and (range_m is None or depth_m is None):
            continue

        box = o.get("box")
        near_k = o.get("near_k")

        base = (
            f"  obj[{i}]"
            + f" label={label}"
            + f" score={_fmt_float(score, 2)}"
            + (f" roi={roi}" if roi is not None else "")
            + f" pts_box={pts_box}"
            + (f" pts_center={pts_center}" if show_center else "")
            + (f" near_k={near_k}" if near_k is not None else "")
            + f" range_m={_fmt_float(range_m, 3)}"
            + f" depth_m={_fmt_float(depth_m, 3)}"
            + (f" height_m={_fmt_float(height_m, 3)}" if height_m is not None else "")
            + f" box={box}"
        )

        if show_center:
            center_px = o.get("center_px")
            center_radius_px = o.get("center_radius_px")
            base += f" center_px={center_px} center_r_px={_fmt_float(center_radius_px, 1)}"

        print(base)

        if pretty:
            print(json.dumps(o, ensure_ascii=False, indent=2))


class DetectionDistanceParser(Node):
    def __init__(
        self,
        topic: str,
        *,
        min_score: float,
        min_points_box: int,
        min_points_center: int,
        require_distance: bool,
        only_roi: Optional[str],
        labels: Optional[Set[int]],
        show_center: bool,
        pretty: bool,
        show_all_keys: bool,
        once: bool,
    ) -> None:
        super().__init__("detection_distance_parser")
        self._cfg = {
            "min_score": min_score,
            "min_points_box": min_points_box,
            "min_points_center": min_points_center,
            "require_distance": require_distance,
            "only_roi": only_roi,
            "labels": labels,
            "show_center": show_center,
            "pretty": pretty,
            "show_all_keys": show_all_keys,
        }
        self._once = once

        self.create_subscription(String, topic, self._on_msg, 10)
        self.get_logger().info(f"Subscribed: {topic}")

    def _on_msg(self, msg: String) -> None:
        try:
            payload: Dict[str, Any] = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"JSON parse failed: {e}")
            return
        _print_payload(payload, **self._cfg)

        if self._once:
            raise SystemExit(0)


def _run_stdin_mode(args: argparse.Namespace) -> int:
    labels = _parse_labels(args.labels) if args.labels else None
    for raw in sys.stdin:
        raw = raw.rstrip("\n")
        payload_str = _extract_json_from_echo_line(raw) or raw.strip()
        if not payload_str:
            continue
        if payload_str == "---":
            continue
        try:
            payload: Dict[str, Any] = json.loads(payload_str)
        except Exception:
            continue
        _print_payload(
            payload,
            min_score=args.min_score,
            min_points_box=args.min_points_box,
            min_points_center=args.min_points_center,
            require_distance=args.require_distance,
            only_roi=args.only_roi,
            labels=labels,
            show_center=args.show_center,
            pretty=args.pretty,
            show_all_keys=args.show_all_keys,
        )
        if args.once:
            return 0
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/camera_4/detection_distance")
    ap.add_argument("--stdin", action="store_true", help="从标准输入读取 JSON/ros2 echo 输出并解析")
    ap.add_argument("--once", action="store_true", help="收到一帧后退出")
    ap.add_argument("--min-score", type=float, default=0.0, help="只打印 score >= N 的目标")
    ap.add_argument("--min-points-box", type=int, default=0, help="只打印 points_in_box >= N 的目标")
    ap.add_argument("--min-points-center", type=int, default=0, help="只打印 points_in_center >= N 的目标")
    ap.add_argument("--require-distance", action="store_true", help="只打印 range_m/depth_m 非空的目标")
    ap.add_argument("--only-roi", choices=["center_circle", "box"], default=None, help="只打印指定 ROI 的结果")
    ap.add_argument("--labels", default=None, help="只打印指定 label，逗号分隔，例如: 0,6,11")
    ap.add_argument("--show-center", action="store_true", help="打印中心点/中心圆/points_in_center 信息")
    ap.add_argument("--pretty", action="store_true", help="额外以缩进 JSON 打印每个 object")
    ap.add_argument("--show-all-keys", action="store_true", help="打印 payload 的所有键名")
    args = ap.parse_args(argv)

    if args.stdin:
        return _run_stdin_mode(args)

    labels = _parse_labels(args.labels) if args.labels else None
    rclpy.init()
    node = DetectionDistanceParser(
        args.topic,
        min_score=args.min_score,
        min_points_box=args.min_points_box,
        min_points_center=args.min_points_center,
        require_distance=args.require_distance,
        only_roi=args.only_roi,
        labels=labels,
        show_center=args.show_center,
        pretty=args.pretty,
        show_all_keys=args.show_all_keys,
        once=args.once,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

