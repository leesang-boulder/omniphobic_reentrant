# Imports for computing, fitting, and plotting
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

# Linear Interpolation of Contact Angle:Surface Tension for EtOH:H2O on FAS-16
def ica(g):
    #theta = np.radians(0.8 * g + 52.4) # Linear Interpolation
    theta = np.arccos((21.67 / g) - 0.64) # Theoretical Interpolation
    return theta

# Cylindrical Pore Liquid Entry Pressure (Young-Laplace model)
def LEP_y(g, r):
    o = ica(g)
    lep = (-10 * g * 2 / r) * np.cos(o)
    return lep 

# Toroidal Pore Liquid Entry Pressure (Purcell model)
def LEP_t(g, r, R):
    o = ica(g)
    sin_arg = np.clip(R * np.sin(o) / (r + R), -1, 1)
    arcsin_term = np.arcsin(sin_arg)
    cos_term1 = np.cos(np.pi - arcsin_term)
    cos_term2 = np.cos(o - np.pi + arcsin_term)
    numerator = cos_term1
    denominator = 1 + (R / r) * (1 - cos_term2)
    lep = - (2 * 10 * g / r) * (numerator / denominator)
    return lep

# Cone-Shaped Pore Liquid Entry Pressure
def LEP_c(g, r, a):
    o = ica(g)
    lep = (-10 * g * 2 / r) * np.cos(o + np.radians(a))
    return lep

# Dictionary mapping model names to functions
models = {
    "cylindrical": LEP_y,
    "toroidal": LEP_t,
    "conical": LEP_c
}

def fit_and_plot(model_name, data_x, data_y, p_guess, p_range, plot_title="Liquid Entry Pressure", save_fig=False):
    # Select the model function
    model = models.get(model_name)
    
    if model is None:
        raise ValueError(f"Model '{model_name}' not found. Choose from: {list(models.keys())}")
    
    # Fit the model to the data
    popt, pcov = curve_fit(model, data_x, data_y, p0=p_guess, bounds=p_range)

    # Calculate R^2
    y_fit_data = model(data_x, *popt)
    ss_res = np.sum((data_y - y_fit_data) ** 2)
    ss_tot = np.sum((data_y - np.mean(data_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return (popt)
    
    # Plotting
    # Generate fitted values for the plot
    #x_range = np.linspace(22, 72, 100)
    #y_fit = model(x_range, *popt)

    #colors = sns.color_palette("colorblind")
    #plt.figure(figsize=(8, 8))
    
    # Plot experimental data
    #plt.plot(data_x, data_y, 'o', label="Experimental Data", color=colors[0])
    
    # Plot fitted curve
    #plt.plot(x_range, y_fit, '-', label=f"Fit: {model_name.capitalize()} Model\nr²={r_squared:.2f}\nParameters ={popt}", color=colors[1])
    
    # Plot styling
    #plt.xlabel("Surface Tension (mN/m)", fontsize=12)
    #plt.ylabel("Pressure (Bar)", fontsize=12)
    #plt.title(plot_title, fontsize=14)
    #plt.legend(fontsize=10)
    
    # Show and save
    #if save_fig:
    #    plt.savefig(f'{model_name}_fitting.png')
    
    #plt.show()

# Experimental data
AAO_data_x = np.array([43, 56, 72])
AAO_data_y = np.array([4.83, 14.48, 24.45])

SiNP_data_x = np.array([31, 43, 56])
SiNP_data_y = np.array([5.18, 15.52, 24.48])

# Fit with all models
p_AAO_cylindrical = fit_and_plot("cylindrical", AAO_data_x, AAO_data_y, p_guess=[20], p_range=([0],[100]), plot_title="Cylindrical Pore Model Fit AAO", save_fig=False)
p_AAO_conical = fit_and_plot("conical", AAO_data_x, AAO_data_y, p_guess=[20, 0], p_range=([20, -45],[21, 45]), plot_title="Conical Pore Model Fit AAO", save_fig=False)
p_AAO_toroidal = fit_and_plot("toroidal", AAO_data_x, AAO_data_y, p_guess=[2, 11], p_range=([0, 0],[100, 1000]), plot_title="Toroidal Pore Model Fit AAO", save_fig=False)
p_SiNP_cylindrical = fit_and_plot("cylindrical", SiNP_data_x, SiNP_data_y, p_guess=[20], p_range=([0],[100]), plot_title="Cylindrical Pore Model Fit SiNP", save_fig=False)
p_SiNP_conical = fit_and_plot("conical", SiNP_data_x, SiNP_data_y, p_guess=[20, 0], p_range=([19, -45],[20, 45]), plot_title="Conical Pore Model Fit SiNP", save_fig=False)
p_SiNP_toroidal = fit_and_plot("toroidal", SiNP_data_x, SiNP_data_y, p_guess=[2, 11], p_range=([0, 0],[100, 1000]), plot_title="Toroidal Pore Model Fit SiNP", save_fig=False)

# Discretize domain and generate fit lines
x_values = np.linspace(22, 72, 100)
AAO_fit_y  = LEP_y(x_values, *p_AAO_cylindrical)
AAO_fit_c  = LEP_c(x_values, *p_AAO_conical)
AAO_fit_t  = LEP_t(x_values, *p_AAO_toroidal)
SiNP_fit_y  = LEP_y(x_values, *p_SiNP_cylindrical)
SiNP_fit_c  = LEP_c(x_values, *p_SiNP_conical)
SiNP_fit_t  = LEP_t(x_values, *p_SiNP_toroidal)

# Plot experimental data 
colors = sns.color_palette("colorblind")
plt.figure(figsize=(8, 8))
plt.plot(AAO_data_x, AAO_data_y, '^', label="Pristine Membrane", color=colors[3])
plt.plot(SiNP_data_x, SiNP_data_y, 's', label="NP-Modified Membrane", color=colors[0])

# Plot fit lines 
plt.plot(x_values, AAO_fit_y, '-', label="Pristine Cylindrical (r=%.2f))" % p_AAO_cylindrical, color=colors[3])
plt.plot(x_values, SiNP_fit_y, '-', label="NP-Modified Cylindrical (r=%.2f))" % p_SiNP_cylindrical, color=colors[0])
plt.plot(x_values, AAO_fit_c, '--', label="Pristine Conical (r=%.2f, a=%.2f))" % (p_AAO_conical[0], p_AAO_conical[1]), color=colors[3])
plt.plot(x_values, SiNP_fit_c, '--', label="NP-Modified Conical (r=%.2f, a=%.2f))" % (p_SiNP_conical[0], p_SiNP_conical[1]), color=colors[0])
plt.plot(x_values, AAO_fit_t, ':', label="Pristine Toroidal (r=%.2f, R=%.2f))" % (p_AAO_toroidal[0], p_AAO_toroidal[1]), color=colors[3])
plt.plot(x_values, SiNP_fit_t, ':', label="NP-Modified Toroidal (r=%.2f, R=%.2f))" % (p_SiNP_toroidal[0], p_SiNP_toroidal[1]), color=colors[0])

# Plot Style
plt.xlabel("Surface Tension (mN/m)", fontsize=12)
plt.ylabel("Pressure (Bar)", fontsize=12)
plt.title("Liquid Entry Pressure", fontsize=14)
plt.legend(fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Show and save
plt.savefig('fit_lines.png')
plt.show()
