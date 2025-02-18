#!/usr/bin/env python3
"""
Extract Points from an EPS File and Save as a TXT/CSV File

This script reads an EPS file (plain-text PostScript), searches for lines that contain
two numbers followed by a "p" command (which we assume indicates a point), extracts the
(x, y) coordinates, applies a scaling factor if needed, and writes the points to an output
file. Each line in the output file contains the x and y values separated by a space.

Usage examples:
    python eps2points.py --input input.eps --output points.txt
    python eps2points.py --input input.eps --output points.csv --scale 512
"""

import argparse
import re

def extract_points_from_eps(eps_path, scale=1.0):
    """
    Extracts (x, y) coordinates from an EPS file.
    
    Parameters:
      eps_path : str
          Path to the EPS file.
      scale    : float, optional
          A scaling factor to apply to the extracted coordinates (default is 1.0).
          For example, if the EPS file contains a "gsave 512 512 scale" command, you
          may want to set scale=512.
    
    Returns:
      points : list of tuples (x, y)
    """
    points = []
    # Regular expression pattern:
    #   - looks for two numbers (optionally with a decimal point)
    #   - optionally separated by whitespace
    #   - followed by whitespace and the letter "p" (possibly with surrounding spaces)
    pattern = re.compile(r"([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+p\b")
    
    with open(eps_path, "r") as file:
        for line in file:
            # Check if the line matches the pattern
            match = pattern.search(line)
            if match:
                x_str, y_str = match.groups()
                try:
                    x = float(x_str) * scale
                    y = float(y_str) * scale
                    points.append((x, y))
                except ValueError:
                    continue
    return points

def write_points_to_file(points, output_path):
    """
    Writes the list of points to a file.
    
    Each line in the output file will have the format:
         x y
    with a space separating the coordinates.
    """
    with open(output_path, "w") as f:
        for pt in points:
            f.write(f"{pt[0]} {pt[1]}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Extract points from an EPS file and save them to a TXT/CSV file."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input EPS file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output text/CSV file.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scaling factor for the coordinates (default=1.0).")
    args = parser.parse_args()
    
    points = extract_points_from_eps(args.input, scale=args.scale)
    write_points_to_file(points, args.output)
    print(f"Extracted {len(points)} points and saved to {args.output}")

if __name__ == "__main__":
    main()
