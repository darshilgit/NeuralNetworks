# Parikh, Darshil
# 1001-55-7968
# 2018-10-28
# Assignment-04-01
import Parikh_04_02
import Parikh_04_03
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import glob
from random import shuffle
#import Parikh_04_03
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.backends.tkagg as tkagg
import scipy.misc

class MainWindow(tk.Tk):
    """
    This class creates and controls the main window frames and widgets
    Farhad Kamangar 2018_06_03
    """

    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.rowconfigure(0, weight=1, minsize=500)
        self.columnconfigure(0, weight=1, minsize=500)
        self.master_frame.rowconfigure(2, weight=10, minsize=400, uniform='xx')
        self.master_frame.rowconfigure(3, weight=1, minsize=10, uniform='xx')
        self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')
        self.master_frame.columnconfigure(1, weight=1, minsize=200, uniform='xx')
        self.left_frame = tk.Frame(self.master_frame)
        self.left_frame.grid(row=2, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.display_activation_functions = LeftFrame(self, self.left_frame, debug_print_flag=self.debug_print_flag)

class LeftFrame:
    """
    This class creates and controls the widgets and figures in the left frame which
    are used to display the activation functions.
    Farhad Kamangar 2018_06_03
    """

    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.number_of_delayed_elements = 10
        self.input_file = 'data_set_1.csv'
        self.alpha = 0.1
        self.training_sample_size = 80
        self.stride = 1
        self.number_of_iterations = 10
        self.xmin = 0
        self.xmax = 100
        self.ymin = 0
        self.ymax = 1
        self.input_weight = None
        self.input_values = []
        self.target_values = []
        self.error = 0
        # self.input_file = ''
        # self.input_values = np.random.uniform(-10, 10, size=(2, 4))
        # self.bias_input = np.ones(4)
        # self.input_values = np.vstack((self.input_values, self.bias_input))
        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1)
        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
        w = self.plot_frame.winfo_screenwidth()
        h = self.plot_frame.winfo_screenheight()
        #self.figure = plt.figure(figsize=[w/100,h/100-2.2])
        self.figure = plt.figure("")
        self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.8])
        # self.axes = self.figure.add_axes()
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Iterations')
        self.axes.set_ylabel('Error')
        # self.axes.margins(0.5)
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # Create a frame to contain all the controls such as sliders, buttons, ...
        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_widget .pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        #########################################################################
        #  Set up the control widgets such as sliders and selection boxes
        #########################################################################
        self.number_of_delayed_elements_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                                   from_=0, to_=100, resolution=1, bg="#DDDDDD",
                                                   activebackground="#FF0000", highlightcolor="#00FFFF", label="Delayed Ele",
                                                   command=lambda event: self.number_of_delayed_elements_slider_callback())
        self.number_of_delayed_elements_slider.set(self.number_of_delayed_elements)
        self.number_of_delayed_elements_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_delayed_elements_slider_callback())
        self.number_of_delayed_elements_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.learning_rate_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                             activebackground="#FF0000", highlightcolor="#00FFFF", label="Alpha",
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.alpha)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.training_sample_size_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0, to_=100, resolution=10, bg="#DDDDDD",
                                             activebackground="#FF0000", highlightcolor="#00FFFF", label="TrainSample %",
                                             command=lambda event: self.training_sample_size_slider_callback())
        self.training_sample_size_slider.set(self.training_sample_size)
        self.training_sample_size_slider.bind("<ButtonRelease-1>", lambda event: self.training_sample_size_slider_callback())
        self.training_sample_size_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.stride_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                               from_=1, to_=100, resolution=1, bg="#DDDDDD",
                               activebackground="#FF0000", highlightcolor="#00FFFF", label="Stride(s)",
                               command=lambda event: self.set_stride_callback())
        self.stride_slider.set(self.stride)
        self.stride_slider.bind("<ButtonRelease-1>", lambda event: self.set_stride_callback())
        self.stride_slider.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        self.number_of_iterations_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                      activebackground="#FF0000", highlightcolor="#00FFFF", label="Iterations",
                                      command=lambda event: self.number_of_iterations_slider_callback())
        self.number_of_iterations_slider.set(self.number_of_iterations)
        self.number_of_iterations_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_iterations_slider_callback())
        self.number_of_iterations_slider.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)

        self.label_for_data_file = tk.Label(self.controls_frame, text="Data File:",
                                                      justify="center")
        self.label_for_data_file.grid(row=0, column=5, sticky=tk.N + tk.E + tk.S + tk.W)
        self.data_file_variable = tk.StringVar()
        self.data_file_dropdown = tk.OptionMenu(self.controls_frame, self.data_file_variable,
                                                          "data_set_1.csv", "data_set_2.csv",
                                                          command=lambda
                                                              event: self.data_file_dropdown_callback())
        self.data_file_variable.set(self.input_file)
        self.data_file_dropdown.grid(row=0, column=6, sticky=tk.N + tk.E + tk.S + tk.W)

        self.adjust_weights_button = tk.Button(self.controls_frame,
                                               text="Adjust Weights(LMS)",
                                               fg="red",
                                               command=lambda: self.adjust_weights_button_callback())

        self.adjust_weights_direct_button = tk.Button(self.controls_frame,
                                                      text="Adjust Weights(Direct)",
                                                      fg="red",
                                                      command=lambda: self.adjust_weights_direct_button_callback())

        self.set_weights_to_zero_button = tk.Button(self.controls_frame,
                                                  text="Set Weights to Zero",
                                                  fg="red",
                                                  command=lambda: self.set_weights_to_zero_button_callback())

        self.adjust_weights_button.grid(row=0, column=7, sticky=tk.N + tk.E + tk.S + tk.W)
        self.adjust_weights_direct_button.grid(row=0, column=8, sticky=tk.N + tk.E + tk.S + tk.W)
        self.set_weights_to_zero_button.grid(row=0, column=9, sticky=tk.N + tk.E + tk.S + tk.W)


        self.canvas.get_tk_widget().bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.canvas.get_tk_widget().bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.left_mouse_release_callback)
        self.canvas.get_tk_widget().bind("<B1-Motion>", self.left_mouse_down_motion_callback)
        self.canvas.get_tk_widget().bind("<ButtonPress-3>", self.right_mouse_click_callback)
        self.canvas.get_tk_widget().bind("<ButtonRelease-3>", self.right_mouse_release_callback)
        self.canvas.get_tk_widget().bind("<B3-Motion>", self.right_mouse_down_motion_callback)
        self.canvas.get_tk_widget().bind("<Key>", self.key_pressed_callback)
        self.canvas.get_tk_widget().bind("<Up>", self.up_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Down>", self.down_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Right>", self.right_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Left>", self.left_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Shift-Up>", self.shift_up_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Shift-Down>", self.shift_down_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Shift-Right>", self.shift_right_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Shift-Left>", self.shift_left_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("f", self.f_key_pressed_callback)
        self.canvas.get_tk_widget().bind("b", self.b_key_pressed_callback)

    def key_pressed_callback(self, event):
        self.root.status_bar.set('%s', 'Key pressed')

    def up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Up arrow was pressed")

    def down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Down arrow was pressed")

    def right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Right arrow was pressed")

    def left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Left arrow was pressed")

    def shift_up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift up arrow was pressed")

    def shift_down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift down arrow was pressed")

    def shift_right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift right arrow was pressed")

    def shift_left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift left arrow was pressed")

    def f_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "f key was pressed")

    def b_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "b key was pressed")

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + 'x=' + str(event.x) + '   y=' + str(
            event.y))
        self.x = event.x
        self.y = event.y
        self.canvas.focus_set()

    def left_mouse_release_callback(self, event):
        self.root.status_bar.set('%s',
                                 'Left mouse button was released. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def left_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_release_callback(self, event):
        self.root.status_bar.set('%s',
                                 'Right mouse button was released. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def right_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + 'x=' + str(event.x) + '   y=' + str(
            event.y))
        self.x = event.x
        self.y = event.y

    ####################

    def get_data(self):
        self.input_weight = np.zeros(shape=((2 * (self.number_of_delayed_elements + 1)) + 1,))
        self.input_values = []
        self.target_values = []
        price_change, volume_change = Parikh_04_03.read_csv_as_matrix(self.input_file)
        start_index = 0
        current_index = self.number_of_delayed_elements
        target_index = current_index + 1
        bias = 1
        while current_index < len(price_change) - 1:
            data = list(price_change[start_index:current_index+1])
            data.extend(list(volume_change[start_index:current_index+1]))
            data.extend([bias,])
            self.input_values.append(data)
            self.target_values.append(price_change[target_index])
            current_index += self.stride
            target_index += self.stride
            start_index += self.stride
        # print(np.array(self.target_values).shape)
        # print(np.array(self.input_values).shape)

    # self.focus_set()
    def display_activation_function(self):
        # self.get_data()
        self.mse, self.mae = Parikh_04_02.calculate_activation_function(self.input_weight, self.alpha,
            self.input_values, self.target_values, self.training_sample_size, self.number_of_iterations, self.learn_type)

        self.axes.cla()
        self.axes.set_xlabel('Iterations')
        self.axes.set_ylabel('Error')
        self.axes.plot(range(1, self.number_of_iterations + 1), self.mse, 'r-', label = 'MSE')
        self.axes.plot(range(1, self.number_of_iterations + 1), self.mae, 'y-',label = 'MAE')
        self.axes.xaxis.set_visible(True)
        # plt.xlim(self.xmin, self.xmax)
        # plt.ylim(self.ymin, self.ymax)
        self.axes.legend()
        plt.title("Mean Squared Error and Max Absolute Error")
        self.canvas.draw()

    def learning_rate_slider_callback(self):
        self.alpha = np.float(self.learning_rate_slider.get())

    def adjust_weights_button_callback(self):
        self.learn_type = 0
        self.display_activation_function()

    def data_file_dropdown_callback(self):
        self.input_file = self.data_file_variable.get()
        self.get_data()

    def set_weights_to_zero_button_callback(self):
        self.input_weight = np.zeros(shape=((2 * (self.number_of_delayed_elements + 1)) + 1,))

    def adjust_weights_direct_button_callback(self):
        self.learn_type = 1
        self.display_activation_function()

    def number_of_delayed_elements_slider_callback(self):
        self.number_of_delayed_elements = self.number_of_delayed_elements_slider.get()
        self.get_data()
    def training_sample_size_slider_callback(self):
        self.training_sample_size = self.training_sample_size_slider.get()

    def set_stride_callback(self):
        self.stride = self.stride_slider.get()

    def number_of_iterations_slider_callback(self):
        self.number_of_iterations = self.number_of_iterations_slider.get()





def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()


main_window = MainWindow(debug_print_flag=False)
# main_window.geometry("500x500")
main_window.wm_state('zoomed')
main_window.title('Assignment_04 --  Parikh')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()
