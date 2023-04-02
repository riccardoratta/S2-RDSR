import os.path as p
import argparse
from time import time

import rasterio
import torch

from rasterio.windows import Window

from models.RDSR import RDSR_20, RDSR_60

from helpers.log import error
from helpers.dataset import norm_SR20, norm_SR60
from helpers.splitter import to_batch, to_image
from helpers.dataset import normalize_t, denormalize_t

parser = argparse.ArgumentParser(
    description="Use the RDSR deep neural network",
)

parser.add_argument(
    "path",
    metavar="P",
    help="Path to a Sentinel-2 tile",
)

parser.add_argument(
    "--x",
    type=int,
    help="Coordinate x",
)

parser.add_argument(
    "--y",
    type=int,
    help="Coordinate x",
)

parser.add_argument(
    "--size",
    type=int,
    help="Size of the patch to SR",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size for inference",
)

PATCH_SIZE = 120

if __name__ == "__main__":
    args = parser.parse_args()

    if not p.isfile(args.path):
        error(f"Input path {args.path} is not valid.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Target device: {device}")

    # SR20 normalization
    norm_t_10m, norm_t_20m = (
        normalize_t(norm_SR20, "10"),
        normalize_t(norm_SR20, "20"),
    )
    d_norm_t_20m = denormalize_t(norm_SR20, "20")

    # SR20 normalization
    norm_t_060m, norm_t_120m, norm_t_360m = (
        normalize_t(norm_SR60, "060"),
        normalize_t(norm_SR60, "120"),
        normalize_t(norm_SR60, "360"),
    )
    d_norm_t_360m = denormalize_t(norm_SR60, "360")

    size_10m = args.size
    size_20m = args.size // 2
    size_60m = args.size // 6

    with rasterio.open(args.path) as image:
        img_10m = rasterio.open(image.subdatasets[0])
        img_20m = rasterio.open(image.subdatasets[1])
        img_60m = rasterio.open(image.subdatasets[2])

        x, y = args.x, args.y

        window = Window(x // 1, y // 1, size_10m, size_10m)  # type: ignore

        p_10m = torch.tensor(
            img_10m.read(
                (1, 2, 3, 4),
                window=window,
                out_dtype="int16",
            )
        )
        p_20m = torch.tensor(
            img_20m.read(
                (1, 2, 3, 4, 5, 6),
                window=Window(x // 2, y // 2, size_20m, size_20m),  # type: ignore
                out_dtype="int16",
            )
        )
        p_60m = torch.tensor(
            img_60m.read(
                (1, 2),
                window=Window(x // 6, y // 6, size_60m, size_60m),  # type: ignore
                out_dtype="int16",
            )
        )

        # save for georeferencing
        crs = img_10m.profile["crs"]
        transform = img_10m.window_transform(window)

        img_10m.close()
        img_20m.close()
        img_60m.close()

        border = 6

        print("Super-resolving 20 meters..", end=" ", flush=True)

        # load 20 model
        model_SR20 = RDSR_20()
        model_SR20.load_state_dict(
            torch.load("model.20.pt", map_location=device)["model"]
        )

        model_SR20.eval()

        start = time()

        # apply model on patch at 10 and 20 meters
        ys = []
        for x_10m, x_20m in zip(
            to_batch(p_10m, PATCH_SIZE // 1, border // 1, args.batch_size),
            to_batch(p_20m, PATCH_SIZE // 2, border // 2, args.batch_size),
        ):
            with torch.no_grad():
                ys.append(
                    d_norm_t_20m(model_SR20(norm_t_10m(x_10m), norm_t_20m(x_20m)))
                )

        print(f"done in {time() - start:.2f}s")

        # generate final image from inference patches
        y_20m = to_image(
            torch.cat(ys, 0),
            120,
            border,
            [p_20m.shape[0], p_10m.shape[1], p_10m.shape[2]],
        )

        print("Super-resolving 60 meters..", end=" ", flush=True)

        # load 60 model
        model_SR60 = RDSR_60()
        model_SR60.load_state_dict(
            torch.load("model.60.pt", map_location=device)["model"]
        )

        model_SR60.eval()

        border = 12

        start = time()

        # apply model on patch at 10, 20 and 60 meters
        ys = []
        for x_060m, x_120m, x_360m in zip(
            to_batch(p_10m, PATCH_SIZE // 1, border // 1, 5),
            to_batch(p_20m, PATCH_SIZE // 2, border // 2, 5),
            to_batch(p_60m, PATCH_SIZE // 6, border // 6, 5),
        ):
            with torch.no_grad():
                ys.append(
                    d_norm_t_360m(
                        model_SR60(
                            norm_t_060m(x_060m),
                            norm_t_120m(x_120m),
                            norm_t_360m(x_360m),
                        )
                    )
                )

        print(f"done in {time() - start:.2f}s")

        # generate final image from inference patches
        y_60m = to_image(
            torch.cat(ys, 0),
            120,
            border,
            [p_60m.shape[0], p_10m.shape[1], p_10m.shape[2]],
        )

        name = f"{p.splitext(p.basename(args.path))[0]}-{args.x}_{args.y}_{args.size}"

        # save output
        with rasterio.open(
            f"{name}.tif",
            "w",
            driver="GTiff",
            height=p_10m.shape[1],
            width=p_10m.shape[2],
            count=12,
            dtype="uint16",
            crs=crs,
            transform=transform,
        ) as dataset:
            dataset.write(y_60m[0], 1)
            dataset.write(p_10m[0], 2)
            dataset.write(p_10m[1], 3)
            dataset.write(p_10m[2], 4)
            dataset.write(y_20m[0], 5)
            dataset.write(y_20m[1], 6)
            dataset.write(y_20m[2], 7)
            dataset.write(p_10m[3], 8)
            dataset.write(y_20m[3], 9)
            dataset.write(y_60m[1], 10)
            dataset.write(y_20m[4], 11)
            dataset.write(y_20m[5], 12)
