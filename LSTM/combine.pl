#!/usr/bin/perl

@all=glob "prediction.dat.*";

foreach $file (@all){
    $i=0;
    open FILE, "$file" or die;
    while ($line=<FILE>){
        chomp $line;
        $sum[$i]+=$line;
        $i++;
    }
    close FILE;
    $file_count++;
}

open NEW, ">prediction.dat" or die;
$imax=$i;
$i=0;
while ($i<$imax){
    $val=$sum[$i]/$file_count;
    print NEW "$val\n";
    $i++;
}
close NEW;

