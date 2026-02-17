# Gnuplot script for bob_results_*.csv files
# Usage: gnuplot plot_bob_multi.gp

set terminal pngcairo size 1200,600 enhanced font 'Arial,12'

set datafile separator ','

set xlabel 'Density'
set xrange [0.0001:1]
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
set key outside top center horizontal samplen 2

# Loop over 5 files
do for [i=0:4] {
    infile = sprintf("bob_results_%d.csv", i)
    outfile = sprintf("timing-bob-%d.png", i)
    
    set output outfile
    
    # Extract parameters from header line
    ref_time = real(system(sprintf("head -1 %s | sed 's/.*ref_time=\\([0-9]*\\).*/\\1/'", infile)))
    blas_time = real(system(sprintf("head -1 %s | sed 's/.*blas_time=\\([0-9]*\\).*/\\1/'", infile)))
    total_mem = real(system(sprintf("head -1 %s | sed 's/.*total_mem=\\([0-9]*\\).*/\\1/'", infile)))
    
    set title sprintf("Bob Attention (run %d): Memory Usage and Time vs Density", i)
    
    plot infile every ::1 using 2:(($6+$7+$8)/1e6) with lines lw 2 lc rgb "blue" title 'Total Memory (K+Q+V)', \
         infile every ::1 using 2:($9/1e6) axes x1y2 with lines lw 2 lc rgb "red" title 'Attention Time', \
         infile every ::1 using 2:($11/1e6) axes x1y2 with lines lw 2 lc rgb "magenta" title 'Attention Dry', \
         (total_mem/1e6) with lines lw 2 dt 2 lc rgb "navy" title 'Dense Memory Baseline', \
         (ref_time/1e6) axes x1y2 with lines lw 2 dt 2 lc rgb "dark-red" title 'Ref Time Baseline', \
         (blas_time/1e6) axes x1y2 with lines lw 2 dt 3 lc rgb "orange" title 'BLAS Time Baseline'
}
