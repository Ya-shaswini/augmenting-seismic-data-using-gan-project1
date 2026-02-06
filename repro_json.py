import numpy as np
from fastapi.encoders import jsonable_encoder
import json

data = {
    "val": np.float32(1.0)
}

try:
    print("Trying to encode np.float32 scalar...")
    encoded = jsonable_encoder(data)
    print("Encoded:", encoded)
except Exception as e:
    print("Scalar failed:", e)

data_list_tolist = {
    "vals": np.array([1.0, 2.0], dtype=np.float32).tolist()
}

try:
    print("\nTrying to encode tolist() result...")
    encoded = jsonable_encoder(data_list_tolist)
    print("Encoded:", encoded)
    print("Types in list:", [type(v) for v in data_list_tolist["vals"]])
except Exception as e:
    print("tolist() failed:", e)
