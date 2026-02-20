#!/usr/bin/sh
perf record -g ./microgpt
perf script > out.perf
stackcollapse-perf.pl out.perf > out.folded
flamegraph.pl out.folded > flamegraph.svg
