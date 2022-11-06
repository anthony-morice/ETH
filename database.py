import sqlite3
import time
from datetime import datetime, timezone
import requests
import sys
import signal
import os

REQUEST_OFFSET = 30 # seconds to offset GET request in fetch_latest

class Database:
  def __init__(self, name, granularity=60, keep_current=False, make_consistent=False):
    self.name = name
    self.granularity = granularity
    self.cb = CoinBase()
    self.keys = self.cb.keys
    self.keep_current = keep_current 
    self.first_fetch = True
    self.make_consistent = make_consistent
    try:
      self.connection = sqlite3.connect(self.name)
    except:
      print(f"Error - Could not connect to database: {self.name}")
      exit(1)
    self.cursor = self.connection.cursor()
    qry = "CREATE TABLE IF NOT EXISTS Data (" +\
          ", ".join([key for key in self.keys]) + ")"
    self.cursor.execute(qry)
    if self.make_consistent and not self.keep_current:
      status, failed = self.establish_consistency()
      if not status:
        print("Error - Could not fill the following gaps:")
        for ts in failed:
          print(f"  {Database.readable_datetime(ts)}")
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
    qry = self.cursor.execute(f"SELECT * FROM Data ORDER BY {self.keys[0]} DESC LIMIT 1")
    return qry.fetchone()

  def get_oldest(self):
    qry = self.cursor.execute(f"SELECT * FROM Data ORDER BY {self.keys[0]} ASC LIMIT 1")
    return qry.fetchone()

  def get_all(self):
    return self.cursor.execute("SELECT * FROM Data").fetchall()

  def truncate(self):
    self.cursor.execute("DELETE FROM Data")

  def query(self, qry):
    self.cursor.execute(qry)
  
  def __handler(self, signum, frame):
    self.__fetch_latest()

  def __fetch_latest(self):
    entry, _ = self.cb.get_candles(int(time.time()), granularity=self.granularity)
    if len(entry) != 0:
      self.insert_many(entry)
      sys.stdout.write("\x1b[0G" + \
      f"{self.name} updated with latest -> {Database.readable_datetime(int(entry[0][0]))}")
      sys.stdout.flush()
    else:
      print("Failed to fetch latest candle - empty GET response")
    if self.first_fetch and self.make_consistent:
      self.first_fetch = False
      status, failed = self.establish_consistency()
      if not status:
        print("Error - Could not fill the following gaps:")
        for ts in failed:
          print(f"  {Database.readable_datetime(ts)}")

  def establish_consistency(self):
    print(f"Attempting to make {self.name} consistent...")
    # find and fill any gaps in data
    missing = []
    last = self.get_last()
    if last is None:
      return True, []
    trail_ts, next_ts = last[0], last[0] - self.granularity
    qry = self.cursor.execute(f"SELECT {self.keys[0]} FROM Data ORDER BY {self.keys[0]} DESC")
    data = qry.fetchall()[1:]
    for ts in data:
      ts = ts[0]
      if ts != next_ts:
        count = (trail_ts - ts) // self.granularity - 1
        missing += [(next_ts, count)]
        trail_ts = ts
        next_ts = ts - self.granularity
    # request missing data
    status = True
    failed = []
    breakpoint()
    for (time_end, count) in missing:
      candles, fail = self.cb.get_candles(time_end, count, self.granularity)
      if len(fail) != 0:
        status = False
        failed += fail
      if len(candles) != 0:
        self.insert_many(candles)
    print("...DONE")
    return status, failed

  def __keep_current(self):
    self.keep_current, self.first_fetch = True, True
    signal.signal(signal.SIGALRM, self.__handler)
    last = self.get_last()
    if last is None: # empty database
      # use next minute aligned timestamp as starting point
      if time.time() % 60 < REQUEST_OFFSET:
        wait = -(time.time() % 60) + 1e-5
      else:
        wait = (60 - time.time() % 60)
    else:
      # determine wait time to keep database timestamp alignment consistent
      # given the existing entries and granularity
      current = int(time.time())
      n = (current - last[0]) // self.granularity + 1
      next_ts = last[0] + self.granularity * n
      wait = next_ts - current
    # note the offset from minute aligned request time
    # this is to avoid empty GET responses from coinbase
    signal.setitimer(signal.ITIMER_REAL, wait + REQUEST_OFFSET, self.granularity)

  def add_historic(self, count):
    # add count historic candles before oldest entry
    last = self.get_oldest()
    if last is None:
      entry, _ = self.cb.get_candles(int(time.time()) - int(time.time()) % 60 - 60, granularity=self.granularity)
      if len(entry) != 0:
        self.insert_many(entry)
      else:
        print("Failed to fetch latest candle add_historic - empty GET response")
        return
      last = self.get_oldest()
    candles, missing = self.cb.request_candles(last[0] - self.granularity, count)
    if len(candles) != 0:
      self.insert_many(candles)
    if len(missing) != count:
      print("Error - the following historic timestamps could not be retrieved:")
      for ts in missing:
        print(f"  {Database.readable_datetime(ts)}")

  def write_to_csv(self):
    filename = f"data_{int(time.time())}.csv"
    try:
      f = open(filename, "w")
      f.write(f"{','.join(self.keys)}\n")
      for row in self.get_all():
        f.write(",".join(list(map(lambda x: str(x), row))) + "\n")
      f.close()
    except:
      print("Error: could not write to file")

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

    Note that a default argument is not used as this is bound at program load.
    Also note that a candle cannot be requested until approximately 30 seconds
    after it is created (coinbase will just give an empty response).
    """
    granularity_options = [60, 300, 900, 3600, 21600, 86400] # in seconds
    if granularity not in granularity_options:
      print(f"Error: granularity of {granularity} is not valid - using {granularity_options[0]}") 
      granularity = granularity_options[0]
    time_end -= time_end % 60 # used to sync time_end with database intervals
    candles = []
    missing = []
    success, skip = True, False
    while num_candles > 0:
      # coinbase api limit of 300 candles per request
      request_size = 300 if num_candles // 300 else num_candles 
      request_candles = self.__get_candles(time_end, request_size, granularity)
      # ensure all requested candles were recieved
      for i, candle in enumerate(request_candles):
        if int(candle[0]) != time_end - i * granularity:
          # missing candle, will try request one more time
          timestamp = time_end - i * granularity
          print(f"Error - {Database.readable_datetime(timestamp)}\
                  candle is absent from requested range. Trying again...")
          skip != success
          if not success: 
            # already failed once so figure out which candles are missing and move on
            missing += timestamp
            continue
          success = False 
          break
      if success or skip:
        success, skip = True, False
        candles += request_candles
        missed = request_size - len(request_candles)
        if len(request_candles) > 0:
          time_end = request_candles[-1][0] - granularity
          num_candles -= len(request_candles)
        if missed != 0:
          print(f"Error - did not receive {missed} candles. Continuing without them...")
          num_candles -= missed
    return candles, missing

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f"USAGE: python3 {sys.argv[0]} <db_name>")
    exit(1)
  db_name = sys.argv[1]
  #db = Database(db_name)
  db = Database(db_name, make_consistent=True)
  db.write_to_csv()
  #db = Database(db_name, keep_current=True)
  while db.keep_current:
    signal.pause()
