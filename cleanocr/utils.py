"""Utility functions"""

def format_time(flt):
  """Pretty prints the timestamp returned by time.time()"""
  
  h = flt//3600
  m = (flt % 3600)//60
  s = flt % 60
  out = []
  if h > 0:
    out.append(str(int(h)))
    if h == 1:
      out.append('hr,')
    else:
      out.append('hrs,')
  if m > 0:
    out.append(str(int(m)))
    if m == 1:
      out.append('min,')
    else:
      out.append('mins,')
  out.append(f'{s:.2f}')
  out.append('secs')
  return ' '.join(out)