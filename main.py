# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.lsb_embed import  generate_stego_dataset

def main():
    # Use a breakpoint in the code line below to debug your script.
    print(f'\n\n\n Deliverable 4: Steganalysis Prototype')  # Press Ctrl+F8 to toggle the breakpoint.
    print(f' Student No. 222117186')  # Press Ctrl+F8 to toggle the breakpoint.

    coverFolder = "data/cover"
    stegoFolder = "data/stego_lsb"

    #validation
    if not os.path.exists(stegoFolder) or len(os.listdir(stegoFolder)) == 0:
        print(f'\nStep 1: Generating LSB  stego images...\n')
        if not os.path.exists(coverFolder) or len(os.listdir(coverFolder)) == 0:
            print(f'ERROR: No cover images found. Please add cover images in data/cover and retry.')
            sys.exit(1)
        generate_stego_dataset(coverFolder, stegoFolder, bpp = 0.4)
    else:
        print(f'Stego images already exists in {stegoFolder}.')

    # print results

    print(f' \nStep 2:  SRM preprocessing running... ')
    #from demo.visualize import demo
    #demo(coverFolder, stegoFolder)

    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
