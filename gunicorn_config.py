# Gunicorn configuration file
workers = 2
timeout = 120  # 2 minutes timeout instead of 30 seconds
bind = "0.0.0.0:10000"
