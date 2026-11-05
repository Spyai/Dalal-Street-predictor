# imports/

Manual drop-in files Dalal reads when a source can't be reached (for example NSE
refusing GitHub's US servers). Everything here except the two files below is
gitignored.

| File | Used for | Where to get it |
|---|---|---|
| `nse_holidays_<year>.csv` (`date,description`) | trading calendar when NSE is unreachable | nseindia.com → Resources → Exchange Communication → Holidays. The 2026 file was assembled from the published calendars of Upstox, Zerodha and StockGro (all agree on the 15 scheduled closures; the 15 Jan election closure is from Zerodha's calendar). Check it against the NSE circular. |
| `ind_<universe>list.csv` (NSE's `Company Name,Industry,Symbol,Series,ISIN Code`) | the stock universe (`UNIVERSE`, default `nifty500`) when NSE is unreachable | niftyindices.com → Indices → Broad Market → Nifty 500 → "Download constituents" |

Screener.in CSV exports (fundamentals, Phase 5) also go here.
