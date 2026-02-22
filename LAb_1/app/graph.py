import matplotlib.pyplot as plt

def plot_elevation_profile(distances, elevations) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(distances, elevations, marker='o')
    plt.title('Distance vs Elevation Profile')
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.grid()
    plt.show()

def plot_cubic_spline_graph(x, y, xm, ym, s, eps) -> None:
    plt.figure(figsize=(10, 6))
    plt.title('Cubic Spline Interpolation of Data Points')

    
    plt.plot(x, y, 'g--', label='Оригінальний профіль (ym)', linewidth=3, alpha=0.4)

    plt.plot(xm, eps, 'r:', label='Похибка ε')

    plt.plot(xm, s, 'b-', label='Кубічний сплайн S(x)', linewidth=1)

    plt.scatter(x, y, color='red', zorder=5, label='Вузли інтерполяції')

    plt.xlabel("Кумулятивна відстань (м)")
    plt.ylabel("Висота (м)")
    plt.legend()
    plt.grid()
    plt.show()