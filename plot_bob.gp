# Gnuplot script for timing-bob.csv
# Usage: gnuplot plot_bob.gp

set terminal pngcairo size 1200,600 enhanced font 'Arial,12'
set output 'timing-bob.png'

set datafile separator ','

# Skip header row
set key autotitle columnhead

set xlabel 'Density'
set xrange [0.001:1]
set logscale x

# Left Y-axis: Memory usage
set ylabel 'Total Memory (MB)' textcolor rgb "blue"
set ytics nomirror textcolor rgb "blue"
set logscale y

# Right Y-axis: Attention time
set y2label 'Attention Time (s)' textcolor rgb "red"
set y2tics nomirror textcolor rgb "red"
set logscale y2

set grid

set tmargin 4
set title 'Bob Attention: Memory Usage and Time vs Density'

set key outside top center horizontal samplen 2

# Plot:
# - Total memory (mem_k + mem_q + mem_v) on left axis (MB)
# - Attention time on right axis (seconds)
# Baselines: memory = 62.67 MB, time = 0.312 s, time = 0.0096 s
plot 'timing-bob.csv' using 2:(($6+$7+$8)/1e6) with lines lw 2 lc rgb "blue" title 'Total Memory (K+Q+V)', \
     '' using 2:($9/1e6) axes x1y2 with lines lw 2 lc rgb "red" title 'Attention Time', \
     (62668800/1e6) with lines lw 2 dt 2 lc rgb "navy" title 'Dense Memory Baseline', \
     (311873/1e6) axes x1y2 with lines lw 2 dt 2 lc rgb "dark-red" title 'Ref Time Baseline', \
     (9622/1e6) axes x1y2 with lines lw 2 dt 3 lc rgb "orange" title 'BLAS Time Baseline'
