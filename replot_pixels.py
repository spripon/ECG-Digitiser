import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def resample_pixels_in_dir(dir, resample_factor):
    for file in tqdm(os.listdir(dir)):
        if file.endswith(".json"):
            file_path = os.path.join(dir, file)

            with open(file_path, "r") as file:
                data = json.load(file)

            leads = data["leads"]
            for i in range(len(leads)):
                pixels = np.array(leads[i]["plotted_pixels"])
                # Use linear interpolation to add more pixels to the plot
                new_pixels = np.zeros(((len(pixels) - 1) * resample_factor, 2))
                for j in range(len(pixels) - 1):
                    new_pixels[j * resample_factor : (j + 1) * resample_factor, 0] = \
                        np.linspace(pixels[j, 0], pixels[j + 1, 0], resample_factor)
                    new_pixels[j * resample_factor : (j + 1) * resample_factor, 1] = \
                        np.linspace(pixels[j, 1], pixels[j + 1, 1], resample_factor)
                data["leads"][i]["dense_plotted_pixels"] = new_pixels.tolist()
                
            if "leads_augmented" in data.keys():
                leads = data["leads_augmented"]
                for i in range(len(leads)):
                    pixels = np.array(leads[i]["plotted_pixels"])
                    # Use linear interpolation to add more pixels to the plot
                    new_pixels = np.zeros(((len(pixels) - 1) * resample_factor, 2))
                    for j in range(len(pixels) - 1):
                        new_pixels[j * resample_factor : (j + 1) * resample_factor, 0] = \
                            np.linspace(pixels[j, 0], pixels[j + 1, 0], resample_factor)
                        new_pixels[j * resample_factor : (j + 1) * resample_factor, 1] = \
                            np.linspace(pixels[j, 1], pixels[j + 1, 1], resample_factor)
                    data["leads_augmented"][i]["dense_plotted_pixels"] = new_pixels.tolist()

            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample plotted pixels in a "
                                                 "directory.")

    parser.add_argument(
        '--resample_factor',
        type=int,
        default=20,
        help='Multiplicative factor for resampling the plotted pixels.'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default="ptb-xl/records500_prepared_w_images",
        help='Directory containing plotted pixels to be resampled.'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Whether to plot the resampled pixels.'
    )

    args = parser.parse_args()

    resample_factor = args.resample_factor
    dir = args.dir
    plot = args.plot

    args = argparse.ArgumentParser()

    resample_pixels_in_dir(dir, resample_factor)
    print("All files saved.")

    if plot:
        for file in os.listdir(dir):
            if file.endswith(".json"):
                file_path = os.path.join(dir, file)

                with open(file_path, "r") as file:
                    data = json.load(file)

                leads = data["leads"]
                for i in range(len(leads)):
                    pixels = np.array(leads[i]["plotted_pixels"])
                    plt.scatter(pixels[:, 1], -pixels[:, 0], s=1)
                plt.figure()
                for i in range(len(leads)):
                    dense_pixels = np.array(leads[i]["dense_plotted_pixels"])
                    plt.scatter(dense_pixels[:, 1], -dense_pixels[:, 0], s=1)
                plt.show()
                break
