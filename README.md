# Image Search

## Requirements

* docker
* docker-compose
* python2.6

## Getting Started

1. run `download-model.sh` to download the caffe image model used
2. `docker-compose up`


## Indexing + Querying

Every image that is queried is also added to the index. As a result,
each image will return itself as a best match, followed by its nearest
neighbors.

To perform a query, POST an image to the `images` end point. See the
`test.py` script for an example.

The query will return matches sorted by distance, containin the matching
images names. The images can be viewed on the same end point `images/image.jpg`


