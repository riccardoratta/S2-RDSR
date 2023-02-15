import os
import os.path as p
import argparse
import shutil

import torch

from helpers.image import Image, downscale
from helpers.sentinel import Bands

parser = argparse.ArgumentParser(
    description="Create training datasets for Sen2-RDSR deep residual network."
)

parser.add_argument("path", metavar="P", help="Path to BigEarthNet dataset")

parser.add_argument(
    "--type",
    choices=["20", "60"],
    nargs="+",
    required=True,
    help='Dataset type, could be "20" or "60" respectively for SR 20m or 60m dataset',
)

parser.add_argument(
    "--patch-extent",
    type=int,
    default=6,
    help="Specify how much BigEarthNet patches to merge when creating the SR60 dataset",
)

parser.add_argument("--output", "-o", default=".", help="Specify the output directory.")

parser.add_argument(
    "--in-place",
    action="store_true",
    default=False,
    help="If during the creation of the dataset remove already used patches from BigEarthNet, use when storage is low.",
)


def merge_bigEarthNet_patch(path: str, patch: str, bands: Bands) -> torch.Tensor:
    """
    Given a path and a name to a BigEarthNet patch, return a single tensor with all the selected
    bands (10m, 20m or 60m), see `Bands` enum for available bands.
    """
    return torch.cat(
        [
            Image(p.join(path, f"{patch}_B{band}.tif")).tensor.float()
            for band in bands.value
        ],
        0,
    )


if __name__ == "__main__":
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.path) or not os.path.isdir(args.path):
        print(f"Input path {args.path} doens't exists.")
        exit(-1)

    args.type = set(args.type)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    SR20_path = p.join(args.output, "SR20")
    SR60_path = p.join(args.output, "SR60")

    if "20" in args.type:
        if os.path.exists(SR20_path):
            print("A SR20 dataset in the output path already exists, abort.")
            exit(-1)
        else:
            os.makedirs(p.join(SR20_path, "20", "eval"))
            os.makedirs(p.join(SR20_path, "40"))

    if "60" in args.type:
        if os.path.exists(SR60_path):
            print("A SR60 dataset in the output path already exists, abort.")
            exit(-1)
        else:
            os.makedirs(p.join(SR60_path, "060", "eval"))
            os.makedirs(p.join(SR60_path, "120"))
            os.makedirs(p.join(SR60_path, "360"))

    if args.in_place is True and len(args.type) != 1:
        print("In-place mode can be used with only one dataset type.")
        exit(-1)

    listdir = os.listdir(args.path)
    max = len(listdir)

    print(f"Adding {max} patches..")

    if "20" in args.type:
        downscale2x = downscale(2)
        for i, patch_dir in enumerate(listdir):
            print(f"\rSR20: {i / max * 100:.2f}%", end="", flush=True)

            # extract 10m and 20m bands from a patch directory
            patch_10m = merge_bigEarthNet_patch(
                p.join(args.path, patch_dir), patch_dir, Bands.m10
            )
            patch_20m = merge_bigEarthNet_patch(
                p.join(args.path, patch_dir), patch_dir, Bands.m20
            )

            if args.in_place:
                # if specified, remove the source patch
                shutil.rmtree(p.join(args.path, patch_dir), ignore_errors=True)

            # save ground truth
            torch.save(
                patch_20m, p.join(args.output, SR20_path, "20", "eval", f"{i}.pt")
            )

            # save downsample of 10m and 20m
            torch.save(
                downscale2x(patch_10m), p.join(args.output, SR20_path, "20", f"{i}.pt")
            )
            torch.save(
                downscale2x(patch_20m), p.join(args.output, SR20_path, "40", f"{i}.pt")
            )

        print(f"\rSR20: completed!")
    if "60" in args.type:
        raise NotImplementedError()
        """
        setdir = set(listdir); patches = []
        for _, patch_dir in enumerate(listdir):
            if patch_dir in setdir:
                match = re.search('(.+)_(\d{1,2})_(\d{1,2})$', patch_dir)
                if match is not None:
                    try:
                        x, y = int(match.group(2)), match.group(3)

                        sub_patches = set()

                        for i in range(0, args.patch_extent):
                            for j in range(0, args.patch_extent):
                                adjacent = f'{match.group(1)}_{x+i}_{y+j}'
                                if adjacent in setdir:
                                    sub_patches.add(adjacent)
                        
                        if len(sub_patches) == args.patch_extent ** 2:
                            for sub_patch in sub_patches:
                                patches.remove(sub_patch)
                            patches.append(sub_patches)
                        else:
                            patches.remove(patch_dir)

                    except IndexError as err:
                        print(f'Unable to extract name, x, or y at patch "{patch_dir}"')
                    except ValueError as err:
                        print(f'ValueError at patch "{patch_dir}"')
                        print(err)
        """
        downscale6x = downscale(6)


print("Training dataset(s) generated!")
