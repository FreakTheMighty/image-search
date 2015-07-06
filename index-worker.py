#!/usr/bin/env python
import sys
from rq import Queue, Connection, Worker
from redis import Redis

redis = Redis(host='redis', port=6379)

# Preload libraries
import ImageSearch

# Provide queue names to listen to as arguments to this script,
# similar to rqworker
with Connection(connection=redis):
    qs = Queue(connection=redis)
    print(qs)
    w = Worker(qs)
    w.work()
