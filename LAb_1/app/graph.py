import matplotlib.pyplot as plt

def plot_elevation_profile(distances, elevations) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(distances, elevations, marker='o')
    plt.title('Distance vs Elevation Profile')
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.grid()
    plt.show()