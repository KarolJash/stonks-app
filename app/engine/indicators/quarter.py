import pandas as pd

def quarter_v_quarter(data, df):
    #data = yf.Ticker("AAPL")
    combined = pd.concat([data, df], axis=1)
    
    #
    combined['Surprise(%)'] = combined['Surprise(%)'].shift(periods=1)
    combined['Reported EPS'] = combined['Reported EPS'].shift(periods=1)

    combined = combined.dropna(subset=['Reported EPS'])

    combined['EPS %'] = round((combined['Reported EPS'] - combined['Reported EPS'].shift(periods=4)) / abs(combined['Reported EPS'].shift(periods=4)) * 100, 2)

    df['q1_yoy'] = combined['EPS %']
    df['q2_yoy'] = combined['EPS %'].shift(periods=1)
    df['q3_yoy'] = combined['EPS %'].shift(periods=2)
    df['q4_yoy'] = combined['EPS %'].shift(periods=3)

    df['surprise_q1'] = combined['Surprise(%)']
    df['surprise_q2'] = combined['Surprise(%)'].shift(periods=1)
    df['surprise_q3'] = combined['Surprise(%)'].shift(periods=2)
    df['surprise_q4'] = combined['Surprise(%)'].shift(periods=3)

    df.ffill(inplace=True)