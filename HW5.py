import numpy as np
import pandas as pd

# === Problem 1 ===
def f(t, y):
    return 1 + (y/t) + (y/t)**2

def f_prime(t, y):
    dfd_t = -y / t**2 - 2 * y**2 / t**3
    dfd_y = 1 / t + 2 * y / t**2
    return dfd_t + dfd_y * f(t, y)

def exact_y(t):
    return t * np.tan(np.log(t))

def euler_method(f, y0, t_span, h):
    t0, tn = t_span
    t = np.arange(t0, tn + h, h)
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(len(t) - 1):
        y[i+1] = y[i] + h * f(t[i], y[i])
    return pd.DataFrame({'t': t, 'Euler_Approx': y, 'Exact_Solution': exact_y(t)})

def taylor_order2(f, f_prime, y0, t_span, h):
    t0, tn = t_span
    t = np.arange(t0, tn + h, h)
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(len(t) - 1):
        y[i+1] = y[i] + h * f(t[i], y[i]) + (h**2 / 2) * f_prime(t[i], y[i])
    return pd.DataFrame({'t': t, 'Taylor_Approx': y, 'Exact_Solution': exact_y(t)})

# === Problem 2 ===
def system_rhs(t, u):
    u1, u2 = u
    du1 = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    du2 = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([du1, du2])


def exact_u1(t):
    return 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)

def exact_u2(t):
    return -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)

def runge_kutta_4(f, u0, t_span, h):
    t0, tn = t_span
    t_vals = np.arange(t0, tn + h, h)
    u_vals = np.zeros((len(t_vals), len(u0)))
    u_vals[0] = u0
    for i in range(len(t_vals)-1):
        t, u = t_vals[i], u_vals[i]
        k1 = h * f(t, u)
        k2 = h * f(t + h/2, u + k1/2)
        k3 = h * f(t + h/2, u + k2/2)
        k4 = h * f(t + h, u + k3)
        u_vals[i+1] = u + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t_vals, u_vals

# === Execute Problem 1 ===
print("=== Problem 1: Euler & Taylor Method Comparison ===")
t_span_1 = (1, 2)
y0_1 = 0
h_1 = 0.1

euler_df = euler_method(f, y0_1, t_span_1, h_1)
taylor_df = taylor_order2(f, f_prime, y0_1, t_span_1, h_1)

print("\nEuler Method:")
print(euler_df)

print("\nTaylor Order 2 Method:")
print(taylor_df)

# === Execute Problem 2 ===
print("\n=== Problem 2: Runge-Kutta 4th Order Comparison ===")
u0_2 = [4/3, 2/3]
t_span_2 = (0, 1)
h_values = [0.1, 0.05]

for h in h_values:
    t_vals, u_vals = runge_kutta_4(system_rhs, u0_2, t_span_2, h)
    df = pd.DataFrame({
        't': t_vals,
        'RK4_u1_Approx': u_vals[:, 0],
        'RK4_u2_Approx': u_vals[:, 1],
        'Exact_u1': exact_u1(t_vals),
        'Exact_u2': exact_u2(t_vals)
    })
    
    print(f"\nRunge-Kutta 4th Order (h={h}):")
    
    if h == 0.1:
        
        df['RK4_u1_Approx'] = df['RK4_u1_Approx'].apply(lambda x: f"{x:.6e}".replace('e', '×10^'))
        df['RK4_u2_Approx'] = df['RK4_u2_Approx'].apply(lambda x: f"{x:.6e}".replace('e', '×10^'))
    
    print(df)
