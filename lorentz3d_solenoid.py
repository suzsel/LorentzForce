import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from scipy.integrate import quad_vec
from matplotlib import cm
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# Generate phi values
phi = np.linspace(0, 2 * np.pi, 100)

num_coils = 8  # Increase number of coils to create a more uniform magnetic field
coil_spacing = 0.2  # Decrease spacing to make the field more uniform

# Define the parametric function for a single coil
def l(phi, z_offset):
    return np.array([np.cos(phi), np.sin(phi), z_offset * np.ones(len(phi))])

# Generate all coils
def solenoid_coils(phi, num_coils, coil_spacing):
    coils = []
    for i in range(num_coils):
        coil = l(phi, z_offset=i * coil_spacing)
        coils.append(coil)
    return np.array(coils)

solenoid = solenoid_coils(phi, num_coils, coil_spacing)

# Initialize plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Generate grid points for the magnetic field
x = np.linspace(-2, 2, 10)
xv, yv, zv = np.meshgrid(x, x, x)

# Initialize the total magnetic field components
Bx_total = np.zeros_like(xv)
By_total = np.zeros_like(xv)
Bz_total = np.zeros_like(xv)

# Define symbolic variables
t, x, y, z = smp.symbols('t, x, y, z')

# Loop over each coil and compute its contribution to the magnetic field
for i in range(num_coils):
    lx, ly, lz = solenoid[i][0], solenoid[i][1], solenoid[i][2]

    # Define symbolic parametric function and separation vector for this coil
    l_sym = smp.Matrix([smp.cos(t), smp.sin(t), lz[0]])  # lz[0] is the z_offset for this coil
    r_sym = smp.Matrix([x, y, z])
    sep_sym = r_sym - l_sym

    # Define the integrand for the magnetic field components for this coil
    integrand_sym = smp.diff(l_sym, t).cross(sep_sym) / sep_sym.norm() ** 3

    # Convert the integrand to numerical functions
    dBxdt = smp.lambdify([t, x, y, z], integrand_sym[0])
    dBydt = smp.lambdify([t, x, y, z], integrand_sym[1])
    dBzdt = smp.lambdify([t, x, y, z], integrand_sym[2])

    def get_B(x, y, z):
        return np.array([quad_vec(dBxdt, 0, 2 * np.pi, args=(x, y, z))[0],
                         quad_vec(dBydt, 0, 2 * np.pi, args=(x, y, z))[0],
                         quad_vec(dBzdt, 0, 2 * np.pi, args=(x, y, z))[0]])

    # Compute the magnetic field for this coil
    B_field = get_B(xv, yv, zv)
    Bx, By, Bz = B_field

    # Accumulate the magnetic field from this coil
    Bx_total += Bx
    By_total += By
    Bz_total += Bz

    # Plot the parametric line representing this coil
    ax.plot(lx, ly, lz, color='green', linewidth=2)

# Calculate the magnitude of the total magnetic field vectors
B_magnitude = np.sqrt(Bx_total ** 2 + By_total ** 2 + Bz_total ** 2)
B_magnitude[B_magnitude == 0] = 1  # Prevent division by zero

# Normalize the vectors for quiver plotting (but retain their magnitudes for color mapping)
Bx_normalized = Bx_total / B_magnitude
By_normalized = By_total / B_magnitude
Bz_normalized = Bz_total / B_magnitude

# Create the colormap normalization based on the vector magnitudes
norm = plt.Normalize(B_magnitude.min(), B_magnitude.max())
cmap = cm.viridis
colors = cmap(norm(B_magnitude)).reshape(-1, 4)

# Plot the combined magnetic field vectors with a colormap
quiver = ax.quiver(xv, yv, zv, Bx_normalized, By_normalized, Bz_normalized,
                   length=0.3, normalize=False, color=colors)

# Lorentz force simulation
particle, = ax.plot([], [], [], 'ro', markersize=8)  # Red dot for the particle
trajectory, = ax.plot([], [], [], 'r--', linewidth=1)  # Dashed line for the trajectory

q = 5.0  # Charge of the particle (in arbitrary units)
m = 1.0  # Mass of the particle (in arbitrary units)
v0 = np.array([1.0, 0.0, -1.0])  # Initial velocity of the particle (perpendicular and parallel components)
r0 = np.array([0.0, 0.0, 1.2])  # Initial position of the particle

# Time parameters
dt = 0.01
num_steps = 300

# Initialize arrays to store position data
x_data = np.zeros(num_steps)
y_data = np.zeros(num_steps)
z_data = np.zeros(num_steps)
x_data[0], y_data[0], z_data[0] = r0

# Function to update the position of the particle
def update_position(r, v, dt):
    # Lorentz force: F = q(v x B) and a = F/m = q(v x B)/m
    B_vec = get_B(r[0], r[1], r[2])
    a = q * np.cross(v, B_vec) / m  # Acceleration in 3D
    # Update the velocity: Ensure no change in speed (only direction changes)
    v_new = v + a * dt
    v_new = v_new * np.linalg.norm(v) / np.linalg.norm(v_new)  # Normalize speed to keep it constant

    # Update the position
    r_new = r + v_new * dt

    return r_new, v_new

v = v0
r = r0
def animate(i):
    global r, v
    r, v = update_position(r, v, dt)
    x_data[i], y_data[i], z_data[i] = r
    # Update the particle's position
    particle.set_data([x_data[i]], [y_data[i]])  # Update x and y
    particle.set_3d_properties([z_data[i]])  # Update z

    # Update the trajectory
    trajectory.set_data(x_data[:i + 1], y_data[:i + 1])  # Update x and y for the trajectory
    trajectory.set_3d_properties(z_data[:i + 1])  # Update z for the trajectory
    ax.set_title(f"v = [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}] ")

    return particle, trajectory, ax.title


# Initialize the animation by clearing the data
def init():
    particle.set_data([], [])
    particle.set_3d_properties([])
    trajectory.set_data([], [])
    trajectory.set_3d_properties([])
    return particle, trajectory


# Create the animation
ani = FuncAnimation(fig, animate, init_func=init, frames=num_steps, interval=20, blit=True, repeat=False)


# Button callback function to toggle magnetic field lines
def toggle_quiver(event):
    quiver.set_visible(not quiver.get_visible())
    plt.draw()


# Create a button and set the callback
ax_button = plt.axes([0.1, 0.05, 0.17, 0.075])  # Position of the button
button = Button(ax_button, 'Toggle Field')
button.on_clicked(toggle_quiver)

ax.set_aspect('equal', 'box')


# Set the axis labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Magnetic Field Visualization with Multiple Coils')

# Set the aspect ratio
ax.set_box_aspect([1, 1, 1])

# Add a color bar to show the magnitude of the vectors
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='|B|')

# Display the plot
# ani.save('solenoid2.gif', writer='imagegick', dpi=200)
plt.show()
