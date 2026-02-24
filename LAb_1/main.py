import requests

from app.haversine import get_elevation_profile
from app.graph import plot_elevation_profile, plot_cubic_spline_graph
from app.finding_cubic_splines import tabluate_function, main as finding_cubic_splines_main
from app.open import get_params_for_plot


url = """https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.
164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.1
66777,24.531927|48.167326,24.530884|48.167011,24.530061|48.16
6053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166
128,24.520214|48.165416,24.517170|48.164546,24.514640|48.1634
12,24.512980|48.162331,24.511715|48.162015,24.509462|48.16214
7,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580
,24.500537|48.160250,24.500106"""

def main():
    response = requests.get(url)
    data = response.json()

    results = data["results"]

    n = len(results)
    print("Number of cells:", n)

    print("\nTablation of cells:\n")
    print("№ | Latitude | Longitude | Elevation (m)")

    for i, cell in enumerate(results, start=1):
        print(f"""{i:2d} | {cell["latitude"]:.6f} | {cell["longitude"]:.6f} | {cell["elevation"]:.2f}""")

    distances, elevations = get_elevation_profile(results, n)
    for i in range(len(distances)):
        print(f"{i:2d} | {distances[i]:5.2f} | {elevations[i]:8.2f}")
    # plot_elevation_profile(distances, elevations)

    N = tabluate_function(distances, elevations)
    finding_cubic_splines_main(N)

    plot_cubic_spline_graph(*get_params_for_plot(N))



if __name__ == "__main__":
    main()