# Data

This repo does not ship the dataset. Download it yourself and place the CSV at:

```
data/raw/hotel_bookings.csv
```

`data/raw/` is git-ignored, so the file is never committed.

## Source  

Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets.
*Data in Brief*, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126

Two Portuguese hotels (a city hotel in Lisbon and a resort hotel in the Algarve),
one row per booking, arrivals from July 2015 to August 2017, about 119k rows.
The Kaggle copy used here was prepared by Jesse Mostipak from the TidyTuesday
release (Thomas Mock and Antoine Bichat).

## Licence

The original Data in Brief article and its dataset are published open access
under CC BY 4.0. Also check the licence shown on the Kaggle page, since re-uploads
can carry their own terms. **Verify the licence before redistributing the data
in any form.** Under CC BY 4.0 you must credit the authors (citation above) if
you share the data or anything derived from it.

## Download

Option 1: Kaggle CLI (needs a Kaggle account and an API token in `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d jessemostipak/hotel-booking-demand -p data/raw --unzip
ls data/raw/hotel_bookings.csv
```

Option 2: browser. Go to https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand,
click Download, unzip, and move `hotel_bookings.csv` into `data/raw/`.

Option 3: the same data is published by TidyTuesday (2020-02-11) as `hotels.csv`.
If you use it, rename it to `hotel_bookings.csv` and check the columns match
(`pricing_lab.data.REQUIRED_COLUMNS` lists the ones the code needs).

## Columns used

| Column | Use |
|---|---|
| `hotel` | Market: City Hotel or Resort Hotel |
| `is_canceled` | Net (realised) vs gross (all bookings) demand |
| `lead_time` | Days between booking and arrival; a control |
| `arrival_date_year`, `arrival_date_month`, `arrival_date_day_of_month` | Arrival date, week and seasonality |
| `stays_in_weekend_nights`, `stays_in_week_nights` | Nights, for revenue |
| `adults`, `children`, `babies` | Drop zero-guest rows |
| `market_segment`, `distribution_channel` | Segments and test units |
| `adr` | Average daily rate: the price |

## No data yet?

Run `make analyse-demo` or tick "Use SYNTHETIC demo data" in the app. Demo
outputs are synthetic and labelled as such.
