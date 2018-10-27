# Parikh, Darshil
# 1001-55-7968
# 2018-10-08
# Assignment-03-01
import Parikh_03_02
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
import Parikh_03_03
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
		# set the properties of the row and columns in the master frame
		# self.master_frame.rowconfigure(0, weight=1,uniform='xx')
		# self.master_frame.rowconfigure(1, weight=1, uniform='xx')
		self.master_frame.rowconfigure(2, weight=10, minsize=400, uniform='xx')
		self.master_frame.rowconfigure(3, weight=1, minsize=10, uniform='xx')
		self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')
		self.master_frame.columnconfigure(1, weight=1, minsize=200, uniform='xx')
		# create all the widgets
		# self.menu_bar = MenuBar(self, self.master_frame, background='orange')
		# self.tool_bar = ToolBar(self, self.master_frame)
		self.left_frame = tk.Frame(self.master_frame)
		#self.right_frame = tk.Frame(self.master_frame)
		#self.status_bar = StatusBar(self, self.master_frame, bd=1, relief=tk.SUNKEN)
		# Arrange the widgets
		# self.menu_bar.grid(row=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
		# self.tool_bar.grid(row=1, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
		self.left_frame.grid(row=2, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
		#self.right_frame.grid(row=2, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
		#self.status_bar.grid(row=3, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
		# Create an object for plotting graphs in the left frame
		self.display_activation_functions = LeftFrame(self, self.left_frame, debug_print_flag=self.debug_print_flag)
		# Create an object for displaying graphics in the right frame
		#self.display_graphics = RightFrame(self, self.right_frame, debug_print_flag=self.debug_print_flag)

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
		self.alpha = 0.1
		self.xmin = 0
		self.xmax = 1000
		self.ymin = 0
		self.ymax = 100
		self.input_weight = np.random.uniform(-0.001, 0.001, size=(785, 10))
		# print(self.input_weight)
		self.input_image_vector = []
		self.target_values = []
		self.errors = []
		self.epochs = []
		self.epoch = 0
		self.error_rate = 0
		self.conf_matrix = []
		#self.input_weight_2 = 1.0
		#self.bias = 0.0
		self.activation_type = "Symmetrical Hard Limit"
		self.learning_method = "Filtered Learning"
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
		self.axes.set_xlabel('Epochs')
		self.axes.set_ylabel('Error Rate')
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
		self.learning_rate_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
		                                    from_=0.001, to_=1.0, resolution=0.01, bg="#DDDDDD",
		                                    activebackground="#FF0000", highlightcolor="#00FFFF", label="alpha",
		                                    command=lambda event: self.learning_rate_slider_callback())
		self.learning_rate_slider.set(self.alpha)
		self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
		self.learning_rate_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

		self.adjust_weights_button = tk.Button(self.controls_frame,
											   text="Learn (Adjust Weights)",
											   fg="red",
											   command=lambda: self.adjust_weights_button_callback())
		self.randomize_weights_button = tk.Button(self.controls_frame,
											   text="Randomize Weights",
											   fg="red",
											   command=lambda: self.randomize_weights_button_callback())
		self.display_confusion_matrix_button = tk.Button(self.controls_frame,
										   text="Display Confusion Matrix",
										   fg="red",
										   command=lambda: self.display_confusion_matrix_button_callback())
		self.adjust_weights_button.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
		self.randomize_weights_button.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
		self.display_confusion_matrix_button.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)
		#########################################################################
		#  Set up the frame for drop down selection
		#########################################################################
		self.label_for_activation_function = tk.Label(self.controls_frame, text="Activation Function Type:",
		                                              justify="center")
		self.label_for_activation_function.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)
		self.activation_function_variable = tk.StringVar()
		self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
		                                                  "Linear", "Hyperbolic Tangent", "Symmetrical Hard Limit", command=lambda
				event: self.activation_function_dropdown_callback())
		self.activation_function_variable.set("Symmetrical Hard Limit")
		self.activation_function_dropdown.grid(row=0, column=5, sticky=tk.N + tk.E + tk.S + tk.W)

		self.label_for_learning_method = tk.Label(self.controls_frame, text="Select Learning Method:",
													  justify="center")
		self.label_for_learning_method.grid(row=0, column=6, sticky=tk.N + tk.E + tk.S + tk.W)
		self.learning_method_variable = tk.StringVar()
		self.learning_method_dropdown = tk.OptionMenu(self.controls_frame, self.learning_method_variable,
														  "Filtered Learning", "Delta Rule", "Unsupervised Hebb", command=lambda
				event: self.learning_method_dropdown_callback())
		self.learning_method_variable.set("Filtered Learning")
		self.learning_method_dropdown.grid(row=0, column=7, sticky=tk.N + tk.E + tk.S + tk.W)
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
####################################
	def read_one_image_and_convert_to_vector(self, file_name):
		img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
		img = img.reshape(-1, 1)  # reshape to column vector and return it
		img = (img / 127.5) - 1.0
		img = np.append(img, [1])
		img = img.transpose()
		return img

	# self.focus_set()
	def display_activation_function(self):
		if self.epoch == 0 or self.epoch >= 1000:
			self.epoch = 0
			self.epochs = []
			self.errors = []
			self.setup_values()
			shuffle(self.input_image_vector)
		error_rate, self.input_weight, epoch, epochs, self.conf_matrix = Parikh_03_02.calculate_activation_function(self.input_weight, self.alpha,
							 self.input_image_vector, self.learning_method, self.activation_type, self.epoch)

		self.epoch += epoch
		self.errors.extend(error_rate)
		self.epochs.extend(epochs)
		self.axes.cla()
		self.axes.set_xlabel('Epochs')
		self.axes.set_ylabel('Error Rate')
		self.axes.plot(self.epochs, self.errors, 'ro')
		self.axes.xaxis.set_visible(True)
		plt.xlim(self.xmin, self.xmax)
		plt.ylim(self.ymin, self.ymax)
		plt.title(self.activation_type)
		self.canvas.draw()

	def learning_rate_slider_callback(self):
		self.alpha = np.float(self.learning_rate_slider.get())
		#self.display_activation_function()

	def activation_function_dropdown_callback(self):
		self.activation_type = self.activation_function_variable.get()
		#self.display_activation_function()

	def learning_method_dropdown_callback(self):
		self.learning_method = self.learning_method_variable.get()
		#self.display_activation_function()

	def adjust_weights_button_callback(self):
		self.display_activation_function()

	def randomize_weights_button_callback(self):
		self.input_weight = []
		self.input_weight = np.random.uniform(-0.001, 0.001, size=(785, 10))
		# print(self.input_weight)
		self.display_activation_function()

	def display_confusion_matrix_button_callback(self):
		if len(self.conf_matrix):
			Parikh_03_03.display_numpy_array_as_table(np.array(self.conf_matrix))
####################
	def setup_values(self):
		for file in glob.glob('./Data/*'):
			a = file.split('_')
			self.input_image_vector.append(np.append(self.read_one_image_and_convert_to_vector(file), [int(a[0][-1])]))

def close_window_callback(root):
	if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
		root.destroy()


main_window = MainWindow(debug_print_flag=False)
# main_window.geometry("500x500")
main_window.wm_state('zoomed')
main_window.title('Assignment_03 --  Parikh')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()
