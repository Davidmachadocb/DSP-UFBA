import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

T0 = 1.0                 
beta2 = 1.0              
L_disp = (T0**2) / beta2 
L_max = 10 * L_disp      
frames = 100             

bits = np.array([1, 0, 1]) 
bit_slots = np.array([-5, 0, 5]) 
bit_period = bit_slots[1] - bit_slots[0] 

t = np.linspace(-15, 15, 1000)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'mathtext.fontset': 'stix'
})

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 7), 
                                     sharex=True, 
                                     gridspec_kw={'height_ratios': [1, 3]})

plt.subplots_adjust(hspace=0.1)

# bit sequence
square_y = np.zeros_like(t)

for pos, bit in zip(bit_slots, bits):
    if bit == 1:
        mask = (t >= pos - bit_period/2) & (t < pos + bit_period/2)
        square_y[mask] = 1
    
    #add bit label
    ax_top.text(pos, 1.05, str(bit), 
                ha='center',
                va='bottom',
                fontsize=14, 
                fontweight='bold',
                color='black')

#plot
ax_top.step(t, square_y, where='mid', color='black', linewidth=1.5)
ax_top.fill_between(t, square_y, step='mid', color='gray', alpha=0.2)
ax_top.set_ylabel("Logic")
ax_top.set_ylim(-0.2, 1.35) 
ax_top.set_yticks([0, 1])
ax_top.grid(linestyle=':', alpha=0.5)


line, = ax_bot.plot([], [], lw=2.5, color='#0055AA', label='Received Power')
fill = ax_bot.fill_between([], [], color='#0055AA', alpha=0.2)

# original signal
ref_y = np.zeros_like(t)
for pos, bit in zip(bit_slots, bits):
    if bit == 1:
        ref_y += np.exp(-(t - pos)**2 / (T0**2))

ax_bot.plot(t, ref_y, '--', color='gray', linewidth=1, alpha=0.6,)

ax_bot.set_xlim(t[0], t[-1])
ax_bot.set_ylim(-0.1, 1.5)
ax_bot.set_xlabel(r"Time ($t/T_0$)")
ax_bot.set_ylabel("Intensity")
ax_bot.legend(loc='upper right', frameon=True)
ax_bot.grid(linestyle=':', alpha=0.6)

dist_text = ax_bot.text(0.02, 0.95, '', transform=ax_bot.transAxes, 
                        verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# broad the signal
def calculate_field(z, time_array, pulse_centers, t0, ld):
    broadening = np.sqrt(1 + (z / ld)**2)
    total_field = np.zeros_like(time_array, dtype=complex)
    for center in pulse_centers:
        dt = time_array - center
        gamma_z = (1 - 1j * (z/ld)) / (1 + (z/ld)**2)
        field = (1 / np.sqrt(1 - 1j * (z/ld))) * np.exp(-0.5 * (dt/t0)**2 * gamma_z)
        total_field += field
    return np.abs(total_field)**2

# animation loop
def init():
    line.set_data([], [])
    dist_text.set_text('')
    return line, dist_text

def update(frame):
    z = (frame / frames) * L_max
    y = calculate_field(z, t, bit_slots[bits==1], T0, L_disp)
    
    line.set_data(t, y)
    
    global fill
    fill.remove()
    fill = ax_bot.fill_between(t, y, color='#0055AA', alpha=0.2)
    
    dist_text.set_text(f"Distance: {z/L_disp:.1f} $L_D$")
    return line, dist_text

anim = animation.FuncAnimation(fig, update, frames=frames, 
                               init_func=init, blit=False, interval=50)

anim.save('build/dispersion_effect_labeled.gif', writer='pillow', fps=20)