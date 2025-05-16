import argparse
import os
from netCDF4 import Dataset


def assemble_data(input_dir, output_file):
    nc_files = [f for f in os.listdir(input_dir) if f.endswith(".nc")]
    nc_files.sort()

    samples = [0]
    with Dataset(os.path.join(input_dir, nc_files[0]), "r") as first_nc:
        samples.append(first_nc.dimensions["sample"].size)
        num_times = first_nc.dimensions["time"].size
        try:
            num_channels = first_nc.dimensions["channel"].size
        except:
            num_channels = None
        x_size = first_nc.dimensions["x"].size
        y_size = first_nc.dimensions["y"].size
        dtype = first_nc.variables[nc_files[0].split("_")[0]].dtype

    for nc_file in nc_files[1:]:
        with Dataset(os.path.join(input_dir, nc_file), "r") as nc:
            samples.append(nc.dimensions["sample"].size)

    num_samples = sum(samples)
    for i in range(1, len(samples)):
        samples[i] += samples[i - 1]
    with Dataset(output_file, "w") as out_nc:
        out_nc.createDimension("sample", num_samples)
        out_nc.createDimension("time", num_times)
        if num_channels is not None:
            out_nc.createDimension("channel", num_channels)
        out_nc.createDimension("x", x_size)
        out_nc.createDimension("y", y_size)
        if num_channels is not None:
            out_nc.createVariable(
                nc_files[0].split("_")[0], dtype, ("sample", "time", "channel", "x", "y"), chunksizes=(1, 1, num_channels, x_size, y_size)
            )
        else:
            out_nc.createVariable(
                nc_files[0].split("_")[0], dtype, ("sample", "time", "x", "y"), chunksizes=(1, 1, x_size, y_size)
            )

        for i, nc_file in enumerate(nc_files):
            with Dataset(os.path.join(input_dir, nc_file), "r") as nc:
                print(f"Processing {os.path.join(input_dir, nc_file)}")
                variable = nc.variables[nc_file.split("_")[0]]
                out_nc[nc_file.split("_")[0]][samples[i] : samples[i + 1]] = variable[:]

    print(f"Saved data to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    assemble_data(args.input_dir, args.output_file)
