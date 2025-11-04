import h5py
import numpy as np
import gdstk
from skimage import measure

def h5_to_gds_contour_fast(h5_filename, dataset_name, pixel_size,
                           layer=1, gds_filename="output.gds",
                           threshold=0.5, tile_size=2048):
    """
    Convert a known dataset inside a large HDF5 file into a GDS file using contour detection.
    
    Parameters:
        h5_filename    : str, path to HDF5 file
        dataset_name   : str, name of dataset in HDF5 file (full path if inside a group)
        pixel_size     : float, layout units per pixel
        layer          : int, GDS layer number
        gds_filename   : str, output filename for the GDS
        threshold      : float, binarization threshold for pixels
        tile_size      : int, pixel tiling dimension for chunked processing
    """
    print(f"\nOpening {h5_filename}...")
    with h5py.File(h5_filename, "r") as f:
        dset = f[dataset_name]
        rows, cols = dset.shape
        print(f"Dataset shape: {rows} rows × {cols} cols")
        print(f"Pixel size: {pixel_size}")
        print(f"Processing in tiles of {tile_size}×{tile_size} pixels...")

        lib = gdstk.Library()
        cell = lib.new_cell("H5_CONTOUR")

        total_polygons = 0
        tile_count = 0

        # Loop through tiles
        for row_start in range(0, rows, tile_size):
            row_end = min(row_start + tile_size, rows)
            for col_start in range(0, cols, tile_size):
                col_end = min(col_start + tile_size, cols)
                tile_count += 1

                # Read tile from H5
                tile = dset[row_start:row_end, col_start:col_end]

                # Binarize
                binary_tile = (tile > threshold).astype(np.uint8)
                if np.sum(binary_tile) == 0:
                    continue  # skip empty tiles

                # Find contours
                contours = measure.find_contours(binary_tile, 0.5)

                # Add polygons to GDS cell
                poly_count_tile = 0
                for contour in contours:
                    transformed = []
                    t_rows, _ = binary_tile.shape
                    for y, x in contour:
                        gx = (col_start + x) * pixel_size
                        gy = (rows - (row_start + y)) * pixel_size
                        transformed.append((gx, gy))
                    if len(transformed) >= 3:  # valid polygon
                        poly = gdstk.Polygon(transformed, layer=layer)
                        cell.add(poly)
                        poly_count_tile += 1

                total_polygons += poly_count_tile
                print(f"Tile {tile_count}: {poly_count_tile} polygon(s) added.")

        # Write to GDS
        lib.write_gds(gds_filename)
        print(f"\n✅ Conversion complete!")
        print(f"Tiles processed: {tile_count}")
        print(f"Total polygons: {total_polygons}")
        print(f"GDS file saved as: {gds_filename}")


# ============================ USAGE EXAMPLE ============================

if __name__ == "__main__":
    # <<< EDIT THESE >>>
    h5_filename   = r"C:/Users/BLuttgenau/Documents/OAM_zone_plate_simulation/output/zone_plate_20251103_161100.h5"  # path to HDF5 file
    dataset_name  = "mask"                # name of dataset inside the HDF5
    zone_plate_name = "3foci_OAM_zone_plate_highres_20251103_161100"  # base name for output GDS  file
    pixel_size    = 1.0                          # layout units per pixel
    layer         = 1                            # GDSII layer number
    gds_filename  = f"C:/Users/BLuttgenau/Documents/OAM_zone_plate_simulation/output/{zone_plate_name}_converted.gds"              # output GDS filename
    
    # Run conversion
    h5_to_gds_contour_fast(
        h5_filename, dataset_name, pixel_size,
        layer=layer, gds_filename=gds_filename,
        threshold=0.5, tile_size=2048
    )