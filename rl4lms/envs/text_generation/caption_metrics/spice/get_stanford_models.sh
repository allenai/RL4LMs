#!/usr/bin/env sh
# This script downloads the Stanford CoreNLP models.

# Adapted from:
# https://panderson.me/spice/

CORENLP=stanford-corenlp-full-2015-12-09
SPICELIB=lib

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo $(pwd)
echo "$(dirname "$0")"

echo "Downloading..."

wget http://nlp.stanford.edu/software/$CORENLP.zip

echo "Unzipping..."

unzip $CORENLP.zip -d $SPICELIB/
mv $SPICELIB/$CORENLP/stanford-corenlp-3.6.0.jar $SPICELIB/
mv $SPICELIB/$CORENLP/stanford-corenlp-3.6.0-models.jar $SPICELIB/
rm -f stanford-corenlp-full-2015-12-09.zip

echo "Done."
