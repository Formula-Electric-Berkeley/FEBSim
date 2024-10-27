import molicell  # Assuming molicell.py contains the main logic
import energus 
import open_loop

def main():
    molicell.plot()  # Call the function that generates the plot
    energus.plot()
    open_loop()
    

if __name__ == "__main__":
    main()
