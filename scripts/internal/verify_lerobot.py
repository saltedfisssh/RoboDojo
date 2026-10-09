"""Check the official APIs and native codecs needed by RoboDojo export."""

from importlib.metadata import version
import io
import sys


def main():
    import av
    import h5py
    import imageio.v3  # noqa: F401
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode  # noqa: F401
    from lerobot.datasets.language import language_feature_info
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    import numpy as np
    from torchcodec.decoders import VideoDecoder  # noqa: F401

    assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"
    assert version("lerobot") == "0.6.1", "The exporter requires LeRobot 0.6.1"
    assert "language_persistent" in language_feature_info()
    RGBEncoderConfig(vcodec="h264", crf=20, preset="fast", g=2)
    av.codec.Codec("libx264", "w")
    with h5py.File(io.BytesIO(), "w") as data:
        data["state"] = np.zeros((2, 14), dtype=np.float32)
        assert data["state"].shape == (2, 14)
    for package in ("lerobot", "numpy", "torch", "torchvision", "torchcodec", "h5py", "av", "datasets", "pyarrow"):
        print(f"{package}=={version(package)}")
    print("Export APIs, HDF5 and native video libraries passed; no simulator was loaded.")


if __name__ == "__main__":
    main()
