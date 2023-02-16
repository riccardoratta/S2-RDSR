import os
import re
import os.path as p
import argparse
import shutil

import torch

from typing import List, Set

from helpers.log import error
from helpers.satellite_image import SatelliteImage, downscale
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

parser.add_argument(
    "--patch-size",
    type=int,
    default=120,
    help="Size of the 10m patches.",
)

parser.add_argument(
    "--output",
    "-o",
    default=".",
    help="Output directory, default to the current one",
)

parser.add_argument(
    "--in-place",
    action="store_true",
    default=False,
    help="If during the creation of the dataset remove already used patches from BigEarthNet, use when storage is low",
)

parser.add_argument(
    "--limit",
    type=int,
    help="Limit how many patches to process",
)


def merge_bigEarthNet_patch(path: str, patch: str, bands: Bands) -> torch.Tensor:
    """
    Given a path and a name to a BigEarthNet patch, return a single tensor with all the selected
    bands (10m, 20m or 60m), see `Bands` enum for available bands.
    """
    return torch.cat(
        [
            SatelliteImage(p.join(path, f"{patch}_B{band}.tif")).tensor
            for band in bands.value
        ],
        0,
    )


def find_adjacent_patches(listdir: List[str], patch_extent=3):
    """
    Given a list of directories in the format `S2B_MSIL2A_20170924T093019_33_62`, where the last two
    values idicates the patch coordinate `x` and `y` in the source tile, find, if there are, the
    adjacent patches, up to `patch_extent` value. For instance an adjacent p. could be the one
    ending with `_33_63` or `_34_64` (up to the maximum extent).

    It returnes only full extent, i.e., if extent is 2 and the search finded only 3 patches then the
    set is not returned.
    """
    setdir = set(listdir)
    patches: List[Set[str]] = []
    n = len(listdir)
    for z, dir in enumerate(listdir):
        # if dir was not already removed
        if dir in setdir:

            print(f"\r> {z / n * 100:.2f}%", end="")

            match = re.search("(.+)_([0-9]{1,2})_([0-9]{1,2})$", dir)
            if match is not None:
                try:
                    x = int(match.group(2))
                    y = int(match.group(3))

                    subset: Set[str] = set()

                    # look in extent range
                    for i in range(patch_extent):
                        for j in range(patch_extent):
                            # candidate adjacent patch
                            adjacent = f"{match.group(1)}_{x+i}_{y+j}"
                            # check if actually exists..
                            if adjacent in setdir:
                                subset.add(adjacent)

                    # if I've found a full subset (all adjacent found!)
                    if len(subset) == patch_extent**2:
                        for patch in subset:
                            setdir.remove(patch)
                        patches.append(subset)
                    else:
                        setdir.remove(dir)

                except IndexError as _:
                    print(f'Unable to extract name, x, or y at patch "{patch_dir}"')
                except ValueError as e:
                    print(f'ValueError at patch "{patch_dir}"', e)

    print("\r", end="")

    return patches


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.path) or not os.path.isdir(args.path):
        error(f"Input path {args.path} doens't exists.")

    if args.in_place is True and len(args.type) != 1:
        error("In-place mode can be used with only one dataset type.")

    listdir = sorted(os.listdir(args.path))

    if args.limit is not None:
        print(f"Warning: a limit of {args.limit} patches as been set.")
        listdir = listdir[0 : args.limit]
        if args.in_place is True:
            error("Limit cannot be used with the in-place mode.")

    args.type = set(args.type)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    SR20_path = p.join(args.output, "SR20")
    SR60_path = p.join(args.output, "SR60")

    if "20" in args.type:
        if os.path.exists(SR20_path):
            error("A SR20 dataset in the output path already exists, abort.")
        else:
            os.makedirs(p.join(SR20_path, "20", "eval"))
            os.makedirs(p.join(SR20_path, "40"))

    if "60" in args.type:
        if os.path.exists(SR60_path):
            error("A SR60 dataset in the output path already exists, abort.")
        else:
            os.makedirs(p.join(SR60_path, "060", "eval"))
            os.makedirs(p.join(SR60_path, "120"))
            os.makedirs(p.join(SR60_path, "360"))

    n = len(listdir)

    print(f"Adding {n} patches..")

    if "20" in args.type:
        downscale2x = downscale(2)
        for i, patch_dir in enumerate(listdir):
            print(f"\rSR20: {i / n * 100:.2f}%", end="")

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
                patch_20m.type(torch.float16),
                p.join(args.output, SR20_path, "20", "eval", f"{i}.pt"),
            )

            # save downsample of 10m and 20m
            torch.save(
                downscale2x(patch_10m).type(torch.float16),
                p.join(args.output, SR20_path, "20", f"{i}.pt"),
            )
            torch.save(
                downscale2x(patch_20m).type(torch.float16),
                p.join(args.output, SR20_path, "40", f"{i}.pt"),
            )

        print(f"\rSR20: completed!")

    if "60" in args.type:

        size, extent = args.patch_size, args.patch_extent

        print(f"Finding adjacent patches with extent={extent} ..")

        patches = find_adjacent_patches(listdir, extent)

        n = len(patches)

        print(f"Adding {n} patches..")

        size_10m, size_20m, size_60m = (
            size * extent,
            size * extent // 2,
            size * extent // 6,
        )

        downscale6x = downscale(6)

        for i, patch_set in enumerate(patches):

            patch_10m = torch.zeros((len(Bands.m10.value), size_10m, size_10m))
            patch_20m = torch.zeros((len(Bands.m20.value), size_20m, size_20m))
            patch_60m = torch.zeros((len(Bands.m60.value), size_60m, size_60m))

            print(f"\rSR60: {i / n * 100:.2f}%", end="")

            for j, patch_dir in enumerate(sorted(patch_set)):

                # subpatch coordinates
                x, y = j % extent, j // extent

                # extract 10m, 20m and 60m bands from a patch directory
                subpatch_10m = merge_bigEarthNet_patch(
                    p.join(args.path, patch_dir), patch_dir, Bands.m10
                )
                subpatch_20m = merge_bigEarthNet_patch(
                    p.join(args.path, patch_dir), patch_dir, Bands.m20
                )
                subpatch_60m = merge_bigEarthNet_patch(
                    p.join(args.path, patch_dir), patch_dir, Bands.m60
                )

                if args.in_place:
                    # if specified, remove the source patch
                    shutil.rmtree(p.join(args.path, patch_dir), ignore_errors=True)

                subsize_10m = subpatch_10m.shape[1]
                subsize_20m = subpatch_20m.shape[1]
                subsize_60m = subpatch_60m.shape[1]

                patch_10m[
                    :,
                    (x * subsize_10m) : (x * subsize_10m + subsize_10m),
                    (y * subsize_10m) : (y * subsize_10m + subsize_10m),
                ] = subpatch_10m

                patch_20m[
                    :,
                    (x * subsize_20m) : (x * subsize_20m + subsize_20m),
                    (y * subsize_20m) : (y * subsize_20m + subsize_20m),
                ] = subpatch_20m

                patch_60m[
                    :,
                    (x * subsize_60m) : (x * subsize_60m + subsize_60m),
                    (y * subsize_60m) : (y * subsize_60m + subsize_60m),
                ] = subpatch_60m

            # save ground truth
            torch.save(
                patch_60m.type(torch.float16),
                p.join(args.output, SR60_path, "060", "eval", f"{i}.pt"),
            )

            # save downsample of 10m, 20m and 60m
            torch.save(
                downscale6x(patch_10m).type(torch.float16),
                p.join(args.output, SR60_path, "060", f"{i}.pt"),
            )
            torch.save(
                downscale6x(patch_20m).type(torch.float16),
                p.join(args.output, SR60_path, "120", f"{i}.pt"),
            )
            torch.save(
                downscale6x(patch_60m).type(torch.float16),
                p.join(args.output, SR60_path, "360", f"{i}.pt"),
            )

        print(f"\rSR60: completed!")

    print("Training dataset(s) generated!")
