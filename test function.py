import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_boxcar_surrogate_gradients(width_h_list):
    vth = 1.0
    max_width = max(width_h_list)
    v_mem = np.linspace(vth - 2 * max_width, vth + 2 * max_width, 1000)

    plt.figure(figsize=(15, 9))
    colors = plt.cm.plasma(np.linspace(0, 1, len(width_h_list)))

    for i, width_h in enumerate(width_h_list):
        h_value = 1.0 / (2.0 * width_h)
        cond = np.abs(v_mem - vth) < width_h
        h = np.ones_like(v_mem) * h_value
        du_do = np.where(cond, h, 0)

        plt.plot(v_mem, du_do, label=f"width_h = {width_h}", color=colors[i], linewidth=2.5)

    plt.axvline(x=vth, color='r', linestyle='--', label='Threshold (v_th)')
    plt.title("Boxcar Surrogate Gradients for Various width_h", fontsize=16)
    plt.xlabel("Membrane Potential (v_mem)", fontsize=12)
    plt.ylabel("Surrogate Gradient Value (du_do)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylim(bottom=0)
    plt.show()

def plot_multiple_boxcar_height_fix_surrogate_gradients(width_h_list):
    vth = 1.0
    max_width = max(width_h_list)
    v_mem = np.linspace(vth - 2 * max_width, vth + 2 * max_width, 1000)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.inferno(np.linspace(0, 1, len(width_h_list)))

    for i, width_h in enumerate(width_h_list):
        cond = np.abs(v_mem - vth) < width_h
        h = np.ones_like(v_mem)
        du_do = np.where(cond, h, 0)

        plt.plot(v_mem, du_do, label=f"width_h = {width_h}", color=colors[i], linewidth=2.5)

    plt.axvline(x=vth, color='r', linestyle='--', label='Threshold (v_th)')
    plt.title("Boxcar (Height-Fixed) Surrogate Gradients for Various width_h", fontsize=16)
    plt.xlabel("Membrane Potential (v_mem)", fontsize=12)
    plt.ylabel("Surrogate Gradient Value (du_do)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylim(bottom=0, top=1.2)
    plt.show()

def plot_multiple_triangle_surrogate_gradients(width_h_list):
    vth = 1.0
    max_width = max(width_h_list)
    v_mem = np.linspace(vth - 2 * max_width, vth + 2 * max_width, 1000)

    plt.figure(figsize=(15, 9))
    colors = plt.cm.viridis(np.linspace(0, 1, len(width_h_list)))

    for i, width_h in enumerate(width_h_list):
        distance = np.abs(v_mem - vth)
        h = (1.0 - (distance / width_h)) * (1.0 / width_h)
        cond = distance < width_h
        du_do = np.where(cond, h, 0)

        plt.plot(v_mem, du_do, label=f"width_h = {width_h}", color=colors[i], linewidth=2.5)

    plt.axvline(x=vth, color='r', linestyle='--', label=f'Threshold (v_th = {vth})')
    plt.title("Triangle Surrogate Gradients for Various width_h", fontsize=16)
    plt.xlabel("Membrane Potential (v_mem)", fontsize=12)
    plt.ylabel("Surrogate Gradient Value (du_do)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylim(bottom=0)
    plt.show()

def plot_multiple_triangle_height_fix_surrogate_gradients(width_h_list):
    vth = 1.0
    max_width = max(width_h_list)
    v_mem = np.linspace(vth - 2 * max_width, vth + 2 * max_width, 1000)

    plt.figure(figsize=(15,9))
    colors = plt.cm.cividis(np.linspace(0, 1, len(width_h_list)))

    for i, width_h in enumerate(width_h_list):
        distance = np.abs(v_mem - vth)
        h = (1.0 - (distance / width_h))  # height °íÁ¤
        cond = distance < width_h
        du_do = np.where(cond, h, 0)

        plt.plot(v_mem, du_do, label=f"width_h = {width_h}", color=colors[i], linewidth=2.5)

    plt.axvline(x=vth, color='r', linestyle='--', label='Threshold (v_th)')
    plt.title("Triangle (Height-Fixed) Surrogate Gradients for Various width_h", fontsize=16)
    plt.xlabel("Membrane Potential (v_mem)", fontsize=12)
    plt.ylabel("Surrogate Gradient Value (du_do)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylim(bottom=0, top=1.2)
    plt.show()

def plot_multiple_asy_surrogate_gradients(width_h_list):
    vth = 1.0
    max_width = max(width_h_list)
    v_mem = np.linspace(vth - 2 * max_width, vth + 2 * max_width, 1000)

    plt.figure(figsize=(15, 9))
    colors = plt.cm.cool(np.linspace(0, 1, len(width_h_list)))

    for i, width_h in enumerate(width_h_list):
        slope = 1.0 / (2.0 * width_h ** 2)
        h = slope * (v_mem - (vth - width_h))
        cond = np.abs(v_mem - vth) < width_h
        du_do = np.where(cond, h, 0)

        plt.plot(v_mem, du_do, label=f"asy (width_h = {width_h})", color=colors[i], linewidth=2.5)

    plt.axvline(x=vth, color='r', linestyle='--', label='Threshold (v_th)')
    plt.title("Asymmetric Surrogate Gradients (Scaled)", fontsize=16)
    plt.xlabel("Membrane Potential (v_mem)")
    plt.ylabel("Surrogate Gradient Value (du_do)")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylim(bottom=0)
    plt.show()

def plot_multiple_asy_height_fix_surrogate_gradients(width_h_list):
    vth = 1.0
    max_width = max(width_h_list)
    v_mem = np.linspace(vth - 2 * max_width, vth + 2 * max_width, 1000)

    plt.figure(figsize=(15, 9))
    colors = plt.cm.winter(np.linspace(0, 1, len(width_h_list)))

    for i, width_h in enumerate(width_h_list):
        h = (1 / (2 * width_h)) * (v_mem - vth) + 0.5
        cond = np.abs(v_mem - vth) < width_h
        du_do = np.where(cond, h, 0)

        plt.plot(v_mem, du_do, label=f"asy_height_fix (width_h = {width_h})", color=colors[i], linewidth=2.5)

    plt.axvline(x=vth, color='r', linestyle='--', label='Threshold (v_th)')
    plt.title("Asymmetric Surrogate Gradients (Height-Fix Style)", fontsize=16)
    plt.xlabel("Membrane Potential (v_mem)")
    plt.ylabel("Surrogate Gradient Value (du_do)")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylim(bottom=0)
    plt.show()


width_h_list = [0.1, 0.5, 1.0, 2.0]
# plot_multiple_boxcar_surrogate_gradients(width_h_list)
# plot_multiple_boxcar_height_fix_surrogate_gradients(width_h_list)
# plot_multiple_triangle_height_fix_surrogate_gradients(width_h_list)
# plot_multiple_triangle_surrogate_gradients(width_h_list)
plot_multiple_asy_surrogate_gradients(width_h_list)
# plot_multiple_asy_height_fix_surrogate_gradients(width_h_list)

