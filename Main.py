import molicell  # Assuming molicell.py contains the main logic
import energus 
import open_loop
import accumulator

def main():
    molicell.plot()  # Call the function that generates the plot
    energus.plot()
    # open_loop.simulate(accumulator.Pack())
    

if __name__ == "__main__":
    main()
