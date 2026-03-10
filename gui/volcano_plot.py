import tkinter as tk
from tkinter import filedialog, colorchooser, ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from adjustText import adjust_text

user_inputs = {}

def get_user_input():
    def browse_file():
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            data.set(filename)
            load_columns()

    def load_columns():
        try:
            df = pd.read_csv(data.get())
            columns = list(df.columns)
            x.set(columns[0])
            y.set(columns[1])
            name.set(columns[2])
            update_menu_options(columns)
            name_updated()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load columns: {e}")

    def update_menu_options(columns):
        for menu, var in zip([x_menu, y_menu, name_menu], [x, y, name]):
            menu['menu'].delete(0, 'end')
            for col in columns:
                menu['menu'].add_command(label=col, command=tk._setit(var, col))
        name.trace_add("write", name_updated)  # Update trace to handle changes

    def name_updated(*args):
        try:
            df = pd.read_csv(data.get())
            selected_name_col = name.get()
            update_points_of_interest_list(df[selected_name_col].values)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update points of interest: {e}")

    def update_points_of_interest_list(items):
        pofi_listbox.delete(0, tk.END)
        for item in items:
            pofi_listbox.insert(tk.END, item)

    def choose_color(var):
        color_code = colorchooser.askcolor(title="Choose a color")[1]
        if color_code:
            var.set(color_code)

    def submit():
        try:
            selected_pofi = [pofi_listbox.get(i) for i in pofi_listbox.curselection()]
            global user_inputs
            user_inputs = {
                'title': title.get(),
                'data': pd.read_csv(data.get()),
                'x': x.get(),
                'y': y.get(),
                'name': name.get(),
                'fc_threshold_lower': float(fc_threshold_lower.get()),
                'fc_threshold_upper': float(fc_threshold_upper.get()),
                'sig_threshold': float(sig_threshold.get()),
                'show_labels': show_labels.get(),
                'ns_color': ns_color.get(),
                'ur_color': ur_color.get(),
                'dr_color': dr_color.get(),
                'poi_color': poi_color.get(),
                'pofi': selected_pofi
            }
            plot_volcano()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to submit inputs: {e}")

    root = tk.Tk()
    root.title("User Input")

    # Styling
    style = ttk.Style()
    style.configure("TButton", padding=6)
    style.configure("TLabel", padding=6)
    style.configure("TEntry", padding=6)
    style.configure("TCheckbutton", padding=6)
    style.configure("TOptionMenu", padding=6)

    title = tk.StringVar()
    data = tk.StringVar()
    x = tk.StringVar()
    y = tk.StringVar()
    name = tk.StringVar()
    fc_threshold_lower = tk.StringVar()
    fc_threshold_upper = tk.StringVar()
    sig_threshold = tk.StringVar()
    ns_color = tk.StringVar()
    ur_color = tk.StringVar()
    dr_color = tk.StringVar()
    poi_color = tk.StringVar()
    show_labels = tk.BooleanVar()
    points_of_interest = tk.StringVar()

    input_frame = ttk.Frame(root)
    input_frame.grid(row=0, column=0, sticky="nsew")

    global plot_frame
    plot_frame = ttk.Frame(root)
    plot_frame.grid(row=0, column=1, sticky="nsew")

    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)
    input_frame.columnconfigure(1, weight=1)

    ttk.Label(input_frame, text="Plot Title").grid(row=0, column=0, sticky="w")
    ttk.Entry(input_frame, textvariable=title).grid(row=0, column=1, columnspan=2, sticky="ew")

    ttk.Label(input_frame, text="Data File").grid(row=1, column=0, sticky="w")
    ttk.Entry(input_frame, textvariable=data).grid(row=1, column=1, sticky="ew")
    ttk.Button(input_frame, text="Browse", command=browse_file).grid(row=1, column=2, sticky="ew")

    ttk.Label(input_frame, text="X-axis Column").grid(row=2, column=0, sticky="w")
    x_menu = ttk.OptionMenu(input_frame, x, "")
    x_menu.grid(row=2, column=1, sticky="ew")

    ttk.Label(input_frame, text="Y-axis Column").grid(row=3, column=0, sticky="w")
    y_menu = ttk.OptionMenu(input_frame, y, "")
    y_menu.grid(row=3, column=1, sticky="ew")

    ttk.Label(input_frame, text="Name Column").grid(row=4, column=0, sticky="w")
    name_menu = ttk.OptionMenu(input_frame, name, "")
    name_menu.grid(row=4, column=1, sticky="ew")

    ttk.Label(input_frame, text="FC Threshold Lower").grid(row=5, column=0, sticky="w")
    ttk.Entry(input_frame, textvariable=fc_threshold_lower).grid(row=5, column=1, sticky="ew")

    ttk.Label(input_frame, text="FC Threshold Upper").grid(row=6, column=0, sticky="w")
    ttk.Entry(input_frame, textvariable=fc_threshold_upper).grid(row=6, column=1, sticky="ew")

    ttk.Label(input_frame, text="Significance Threshold").grid(row=7, column=0, sticky="w")
    ttk.Entry(input_frame, textvariable=sig_threshold).grid(row=7, column=1, sticky="ew")

    ttk.Checkbutton(input_frame, text="Show Labels", variable=show_labels).grid(row=8, column=0, columnspan=2, sticky="w")
    
    ttk.Label(input_frame, text="Non-Significant Color").grid(row=9, column=0, sticky="w")
    ttk.Entry(input_frame, textvariable=ns_color).grid(row=9, column=1, sticky="ew")
    ttk.Button(input_frame, text="Choose Color", command=lambda: choose_color(ns_color)).grid(row=9, column=2, sticky="ew")

    ttk.Label(input_frame, text="Upregulated Color").grid(row=10, column=0, sticky="w")
    ttk.Entry(input_frame, textvariable=ur_color).grid(row=10, column=1, sticky="ew")
    ttk.Button(input_frame, text="Choose Color", command=lambda: choose_color(ur_color)).grid(row=10, column=2, sticky="ew")

    ttk.Label(input_frame, text="Downregulated Color").grid(row=11, column=0, sticky="w")
    ttk.Entry(input_frame, textvariable=dr_color).grid(row=11, column=1, sticky="ew")
    ttk.Button(input_frame, text="Choose Color", command=lambda: choose_color(dr_color)).grid(row=11, column=2, sticky="ew")

    ttk.Label(input_frame, text="POI Color").grid(row=12, column=0, sticky="w")
    ttk.Entry(input_frame, textvariable=poi_color).grid(row=12, column=1, sticky="ew")
    ttk.Button(input_frame, text="Choose Color", command=lambda: choose_color(poi_color)).grid(row=12, column=2, sticky="ew")

    ttk.Label(input_frame, text="Points of Interest").grid(row=13, column=0, sticky="w")
    pofi_listbox = tk.Listbox(input_frame, listvariable=points_of_interest, selectmode=tk.MULTIPLE)
    pofi_listbox.grid(row=13, column=1, sticky="ew", columnspan=2)

    ttk.Button(input_frame, text="Submit", command=submit).grid(row=14, column=0, columnspan=3, sticky="ew")

    return root

def check_pofi_in_data(pofi, data, name):
    valid_pofi = []
    if not pofi:
        print("No points of interest selected")
    else:
        for p in pofi:
            if p in data[name].values:
                valid_pofi.append(p)
            else:
                print(f'{p} not found in the data')
    return valid_pofi

def plot_volcano():
    global user_inputs
    data = user_inputs['data']
    x = user_inputs['x']
    y = user_inputs['y']
    name = user_inputs['name']
    fc_threshold_lower = user_inputs['fc_threshold_lower']
    fc_threshold_upper = user_inputs['fc_threshold_upper']
    sig_threshold = user_inputs['sig_threshold']
    show_labels = user_inputs['show_labels']
    ns_color = user_inputs['ns_color']
    ur_color = user_inputs['ur_color']
    dr_color = user_inputs['dr_color']
    poi_color = user_inputs['poi_color']
    pofi = user_inputs['pofi']
    valid_pofi = check_pofi_in_data(pofi, data, name)
    title = user_inputs['title']

    fig, ax = plt.subplots()

    ax.scatter(x=data[x], y=data[y], s=1, label="Not significant", color=ns_color)
    down = data[(data[x] >= fc_threshold_upper) & (data[y] >= sig_threshold)]
    up = data[(data[x] <= fc_threshold_lower) & (data[y] >= sig_threshold)]
    ax.scatter(x=down[x], y=down[y], s=3, label="Up-regulated", color=ur_color)
    ax.scatter(x=up[x], y=up[y], s=3, label="Down-regulated", color=dr_color)

    texts = []
    for i in range(len(data)):
        if data[name][i] in valid_pofi:
            ax.scatter(data[x][i], data[y][i], s=10, color=poi_color)
            texts.append(ax.text(data[x][i], data[y][i], data[name][i], fontsize=8, color='black'))
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.axvline(fc_threshold_lower, color="grey", linestyle="--")
    ax.axvline(fc_threshold_upper, color="grey", linestyle="--")
    ax.axhline(sig_threshold, color="grey", linestyle="--")
    if show_labels:
        ax.legend()
    ax.set_title(title)

    for widget in plot_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    global root
    root = get_user_input()
    root.mainloop()

if __name__ == "__main__":
    main()
