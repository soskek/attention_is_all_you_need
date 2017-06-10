curl -O http://www.statmt.org/europarl/v7/fr-en.tgz
curl -O http://www.statmt.org/wmt15/dev-v2.tgz
gzip -dc fr-en.tgz | tar xvf -
gzip -dc dev-v2.tgz | tar xvf -
rm fr-en.tgz
rm dev-v2.tgz
