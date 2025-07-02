import numpy as np
import matplotlib.pyplot as plt

def plot_asy_area_one(width_h_list):
    vth = 1.0
    max_width = max(width_h_list)
    vmem = np.linspace(vth - 2*max_width, vth + 2*max_width, 1000)

    plt.figure(figsize=(10,6))

    for width_h in width_h_list:
        slope = 1.0 / (2 * width_h ** 2)
        h = slope * (vmem - (vth - width_h))
        cond = np.abs(vmem - vth) < width_h
        du_do = np.where(cond, h, 0)
        plt.plot(vmem, du_do, label=f"width_h={width_h}")

    plt.axvline(vth, color='r', linestyle='--')
    plt.title("Asymmetric Surrogate Gradient (Area = 1)")
    plt.xlabel("v_mem")
    plt.ylabel("du_do")
    plt.grid(True)
    plt.legend()
    plt.show()

plot_asy_area_one([0.1, 0.5, 1.0, 2.0])