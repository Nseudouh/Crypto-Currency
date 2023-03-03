from locale import currency
from pyexpat import model
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

currency_symbol = {'LTC': 0, 'KSM': 1, 'BTC': 2, 'FTM': 3, 'DOGE': 4, 'BNB': 5, 'MKR': 6, 'XMR': 7, 'ETH': 8, 'ADA': 9, 'THETA': 10, 
          'TRX': 11, 'WBTC': 12, 'MATIC': 13, 'LUNA1': 14, 'HNT': 15, 'AVAX': 16, 'AXS': 17, 'LDO': 18}
# create a main window
window = tk.Tk()
window.title('Crypto currency prediction')
frame = tk.Frame(master= window, width= 500, height= 300)
frame.pack()

# Set heading label
heading = tk.Label(master= frame, text= "Welcome to Crypto currency prediction App", bg= 'green', fg= 'black', font= 17)

# function to validate that digits is entered
def validate_input(input_val):
    return input_val.isdigit()

# create open label and entry
open_form = tk.Frame(bg= 'orange', borderwidth= 1)
open_form.pack(fill= tk.BOTH, side= tk.LEFT)
open_validation = open_form.register(validate_input)
open_label = tk.Label(master= open_form, text= "Open price")
open_entry = tk.Entry(master= open_form, width= 5, validate= 'key', validatecommand= (open_validation, '%S'))
open_label.pack()
open_entry.pack()

# create high label and entry
high_label = tk.Label(master= open_form, text= "High price")
high_entry = tk.Entry(master= open_form, width= 5, validate= 'key', validatecommand= (open_validation, '%S'))
high_label.pack()
high_entry.pack()

# create Low label and entry
low_form = tk.Frame(bg= 'orange', borderwidth= 1)
low_form.pack(fill= tk.BOTH, side= tk.RIGHT)
low_validation = open_form.register(validate_input)
low_label = tk.Label(master= low_form, text= "Low price")
low_entry = tk.Entry(master= low_form, width= 5, validate= 'key', validatecommand= (low_validation, '%S'))
low_label.pack()
low_entry.pack()

# create volume label and entry
volume_label = tk.Label(master= low_form, text= "Volume")
volume_entry = tk.Entry(master= low_form, width= 5, validate= 'key', validatecommand= (low_validation, '%S'))
volume_label.pack()
volume_entry.pack()

# create Symbol label and entry
symbol_form = tk.Frame(bg= 'orange', borderwidth= 3)
symbol_form.pack(fill= tk.BOTH, side= tk.TOP)
symbol_validation = open_form.register(validate_input)
symbol_label = tk.Label(master= symbol_form, text= "Currency Symbol")
symbol_entry = tk.Entry(master= symbol_form, width= 5, validate= 'key', validatecommand= (symbol_validation, '%S'))
symbol_label.pack()
symbol_entry.pack()

# Create a callback function
def callback():
    open = float(open_entry.get())
    high = float(high_entry.get())
    low = float(low_entry.get())
    volume = float(volume_entry.get())
    symbol = float(symbol_entry.get())
    if symbol > 18:
        print("Currency not available")
        exit()
    currency_value = [open, high, low, volume, symbol]
    print(currency_value)
    return currency_value

# Function to retrieve currency code
def get_code(entry):
    currency_code = list(currency_symbol.values())
    symbol_code = []
    for code in currency_code:
        if code == entry:
            for symbol in currency_symbol:
                if currency_symbol.get(symbol) == code:
                    symbol_code.append(symbol)
    return symbol_code[0]

# Function to predict currency prices
def prediction(x):
    model = joblib.load('final_model.sav')
    currency_value_data = np.array(x).reshape(1, -1)
    predicted_price = model.predict(currency_value_data)
    symbol = get_code(int(symbol_entry.get()))
    print(f'The predicted price of {symbol} = ', predicted_price[0])
    return predicted_price


# Function to handle click event
def click_handler(event):
    currency_value_data = callback()
    predicted_price = prediction(currency_value_data)
    symbol = get_code(int(symbol_entry.get()))
    messagebox.showinfo(f'Predicted Price of {symbol}', predicted_price[0])


# Create the predict button
button_form = tk.Frame(bg= 'orange', borderwidth= 1)
button_form.pack(fill= tk.BOTH, side= tk.BOTTOM)
predict_button = tk.Button(master= button_form, text= 'Predict', fg= 'black', bg= 'white', width= 10, height= 3)
predict_button.pack()
predict_button.bind('<Button-1>', click_handler)

window.mainloop()


