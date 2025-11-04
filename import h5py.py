import h5py

with h5py.File("C:/Users/BLuttgenau/Documents/OAM_zone_plate_simulation/output/zone_plate_20251103_160720.h5", "r") as f:
    print("\nTop-level datasets and groups:")
    for key in f.keys():
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            print(f"📄 Dataset: {key}, shape={obj.shape}")
        else:
            print(f"📁 Group: {key}")

        # If it's a group, list its contents
        if isinstance(obj, h5py.Group):
            for subkey in obj.keys():
                subobj = obj[subkey]
                if isinstance(subobj, h5py.Dataset):
                    print(f"    📄 Dataset: {key}/{subkey}, shape={subobj.shape}")