# CDC 2000 growth charts (LMS parameters)

- `wtage.csv`: weight-for-age, ages 2-20 years
- `statage.csv`: stature-for-age, ages 2-20 years

Source: Centers for Disease Control and Prevention, National Center for Health Statistics,
https://www.cdc.gov/growthcharts/data/zscore/wtage.csv and statage.csv (retrieved 26 October 2026).
Public-domain US government data. Columns: Sex (1 = male, 2 = female), Agemos (age in months), and
the L, M, S parameters, with which a measurement X has z-score ((X/M)^L - 1) / (L*S).

Used by `src/simulation/growth.py` to add normal growth to the virtual patients aged 12-17.
