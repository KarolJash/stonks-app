import yfinance
import numpy as np
import random

def check_delisted(count):
    arr = np.load('storage/models/delisted.npy')

    start = random.randrange(0,len(arr[:]) - count)

    for ticker in arr[start:start + count]:
        print(ticker)
        try:
            data = yfinance.Ticker(ticker).history(period='5d')

            if (data.empty):
                print(f'Ticker: {ticker} is delisted from NYSE ✅✅✅')
            else:
                print(f'Ticker: {ticker} is still active 🚩🚩🚩')
                arr.remove(ticker)
        except Exception as e:
            print(e)
    
    np.save('storage/models/delisted.npy', arr)

def add_delisted(ticker: str):
    arr = np.load('storage/models/delisted.npy')

    np.save('storage/models/delisted.npy', np.append(arr, ticker))

if __name__ == "__main__":
    check_delisted(input("How many stocks should it look through: "))