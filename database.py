import sqlite3
import time
from datetime import datetime, timezone
import requests
import sys
import signal
import os

class Database:
  def __init__(self, name, granularity=60, keep_current=False):
    self.name = name
    self.granularity = granularity
    self.cb = CoinBase()
    self.keys = self.cb.keys
    self.keep_current = keep_current 
    try:
      self.connection = sqlite3.connect(self.name)
    except:
      print(f"Error - Could not connect to database: {self.name}")
      exit(1)
    self.cursor = self.connection.cursor()
    qry = "CREATE TABLE IF NOT EXISTS Data (" +\
          ", ".join([key for key in self.keys]) + ")"
    self.cursor.execute(qry)
    if self.keep_current:
      self.__keep_current()
  
  def insert(self, row):
    qry = "INSERT INTO Data VALUES(" + ", ".join(['?' for _ in self.keys]) + ")"
    self.cursor.execute(qry, row)
    self.connection.commit()
    
  def insert_many(self, rows):
    qry = "INSERT INTO Data VALUES(" + ", ".join(['?' for _ in self.keys]) + ")"
    self.cursor.executemany(qry, rows)
    self.connection.commit()

  def get_last(self):
    qry = self.cursor.execute("SELECT * FROM Data ORDER BY ? DESC LIMIT 1",\
        (self.keys[0],))
    return qry.fetchone()

  def get_all(self):
    return self.cursor.execute("SELECT * FROM Data").fetchall()

  def truncate(self):
    self.cursor.execute("DELETE FROM Data")

  def query(self, qry):
    self.cursor.execute(qry)
  
  def __handler(self, signum, frame):
    self.fetch_latest()

  def fetch_latest(self):
    entry = self.cb.get_candles(int(time.time()), granularity=self.granularity) # can't use default value
    self.insert_many(entry)
    sys.stdout.write("\x1b[0G" + \
    f"{self.name} updated with latest -> {Database.readable_datetime(int(entry[0][0]))}")
    sys.stdout.flush()

  def establish_consistency(self):
    # starting from oldest entry
    # find any gaps in data through present
    end_time = int(time.time())

  def __keep_current(self):
    self.keep_current = True
    signal.signal(signal.SIGALRM, self.__handler)
    signal.setitimer(signal.ITIMER_REAL, 1e-5, self.granularity) # non zero init value required

  @staticmethod
  def readable_datetime(time_to_convert):
    return time.asctime(time.localtime(time_to_convert))

class CoinBase:
  def __init__(self, product_id = "ETH-USD"):
    self.product_id = "ETH-USD"
    self.keys = ("time", "low", "high", "open", "close", "volume")
  
  def __get_candles(self, time_end, num_candles, granularity):
    if num_candles > 300 or num_candles < 1:
      print("Error: num_candles of {} is not valid - using default of 1".format(num_candles)) 
      num_candles = 1 # 1 - 300 (cannot request more)
    # format start and end timestamps
    # time needs to be in iso 8601 with milliseconds
    time_end -= time_end % granularity
    time_start = time_end - (num_candles - 1) * granularity # query last num_candles candles
    time_end = datetime.fromtimestamp(time_end, tz=timezone.utc
                       ).isoformat(timespec='milliseconds')[:-6] + "Z"
    time_start = datetime.fromtimestamp(time_start, tz=timezone.utc
                         ).isoformat(timespec='milliseconds')[:-6] + "Z" 
    # get and parse candles
    candles = []
    url = "".join(("https://api.exchange.coinbase.com/products/", self.product_id,
                   "/candles?granularity=", str(granularity),
                   "&start=", time_start, "&end=", time_end))
    headers = {"Accept": "application/json"}
    try:
      response = requests.request("GET", url, headers=headers)
    except:
      print(f"Error - GET request failed: {response.text}")
    else:
      try:
        candles = response.text.split("],[")
        for i, candle in enumerate(candles):
          candles[i] = list(map(lambda x: float(x), candle.strip("[]").split(",")))
      except:
        print(f"Error - could not parse GET response: {candles}")
        candles = []
    return candles

  def get_candles(self, time_end, num_candles = 1, granularity = 60):
    """
    @param time_end - of the form seconds since the epoch in UTC (i.e. time.time())
    Note that a default argument is not used as this is bound at program load
    """
    granularity_options = [60, 300, 900, 3600, 21600, 86400] # in seconds
    if granularity not in granularity_options:
      print(f"Error: granularity of {granularity} is not valid - using {granularity_options[0]}") 
      granularity = granularity_options[0]
    time_end -= time_end % 60 # used to sync time_end with database intervals
    candles = []
    success, skip = True, False
    while num_candles > 0:
      # coinbase api limit of 300 candles per request
      request_size = 300 if num_candles // 300 else num_candles 
      request_candles = self.__get_candles(time_end, request_size, granularity)
      # ensure all requested candles were recieved
      for i, candle in enumerate(request_candles):
        if int(candle[0]) != time_end - i * granularity:
          # missing candle, will try request one more time
          print(f"Error - {Database.readable_datetime(time_end - i * granularity)}\
                  candle is absent from requested range. Trying again...")
          skip = !success
          success = False 
          break
      if success or skip:
        success, skip = True, False
        candles += request_candles
        time_end = int(candles[-1][0]) - granularity 
        num_candles -= len(request_candles)
    return candles

  def write_to_csv(self, data,
      filename = "ETH-USD_"):
    filename += f"{str(data[0][0])}_{str(data[-1][0])}.csv"
    try:
      f = open(filename, "w")
      f.write(f"{','.join(self.keys)}\n")
      for row in data:
        f.write(",".join(row) + "\n")
      f.close()
    except:
      print("Error: could not write to file")

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f"USAGE: python3 {sys.argv[0]} <db_name>")
    exit(1)
  db_name = sys.argv[1]
  db = Database(db_name, keep_current=True)
  while db.keep_current:
    signal.pause()
