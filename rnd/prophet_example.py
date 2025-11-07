from neuralprophet import NeuralProphet
import pandas as pd

df = pd.read_csv('../data/qty_daily_company.csv')

m = NeuralProphet()
# show plots correctly in jupyter notebooks
m.set_plotting_backend("plotly-static")
metrics = m.fit(df)


predicted = m.predict(df)
forecast = m.predict(df)

m.plot(forecast)
