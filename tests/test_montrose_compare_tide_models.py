import pandas as pd

from eo_tides.model import model_tides


directory = "/home/mh322u/tide_models_scotland"

lon = -2.45
lat = 56.73

times = pd.date_range(
    "2020-01-01",
    "2020-01-03 23:00",
    freq="1h",
)


print("EOT20")

eot20 = model_tides(
    x=lon,
    y=lat,
    time=times,
    model="EOT20",
    directory=directory,
    output_format="wide",
    crop=True,
    parallel=False,
)


print("FES2022")

fes2022 = model_tides(
    x=lon,
    y=lat,
    time=times,
    model="FES2022",
    directory=directory,
    output_format="wide",
    crop=False,
    parallel=False,
)


print()
print(eot20.head())

print()
print(fes2022.head())